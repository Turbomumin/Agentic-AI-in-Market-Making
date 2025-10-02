from __future__ import annotations
"""
LangChain Market Maker — LLM Advisory or Direct Mode (throttled)
- Integer price units throughout
- Phase-2 tools (inventory, open orders summary, fill-rates, rolling vol) + CSVs
- Groq LLM OFF by default; when ON:
    * advice mode: adjust half-spread and side hint
    * direct mode: LLM returns explicit orders to place/cancel; agent executes
- Performance: LLM called every `lc_every` wakeups, compact prompt, capped tokens, optional timeout
"""

from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Optional, Tuple, List, Dict, Any
import math, os, csv, json, time

import numpy as np
import pandas as pd

from agent.examples.SubscriptionAgent import SubscriptionAgent

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore


@dataclass
class Quote:
    bid_px: Optional[int]
    ask_px: Optional[int]
    qty: int


class LangChainMarketMaker(SubscriptionAgent):
    # ----- robust extractors (unchanged idea) -----
    @staticmethod
    def _attr_get(obj, names: List[str]):
        for n in names:
            try:
                if isinstance(obj, dict) and n in obj and obj[n] is not None:
                    return obj[n]
                v = getattr(obj, n)
                if v is not None:
                    return v
            except Exception:
                continue
        return None

    def _extract_side_any(self, body: dict, oid) -> Optional[str]:
        side = body.get('side')
        if side: return str(side)
        if 'is_buy' in body and body['is_buy'] is not None:
            return 'BUY' if body['is_buy'] else 'SELL'
        order_obj = body.get('order')
        if order_obj is not None:
            v = self._attr_get(order_obj, ['side','is_buy'])
            if isinstance(v, bool): return 'BUY' if v else 'SELL'
            if isinstance(v, str) and v: return v
        if oid in getattr(self, 'order_meta', {}):
            return self.order_meta[oid].get('side')
        return None

    def _extract_price_any(self, body: dict) -> Optional[int]:
        for k in ('fill_price','execution_price','avg_price','price','limit_price','px'):
            v = body.get(k)
            if v is not None:
                try:    return int(v)
                except: 
                    try: return int(round(float(v)))
                    except: pass
        order_obj = body.get('order')
        if order_obj is not None:
            v = self._attr_get(order_obj, ['limit_price','price','px'])
            if v is not None:
                try: return int(v)
                except: pass
        return None

    def _extract_qty_any(self, body: dict) -> Optional[int]:
        for k in ('fill_qty','qty','quantity','filled_quantity','quantity_filled','execution_qty'):
            v = body.get(k)
            if v is not None:
                try: return int(v)
                except: pass
        order_obj = body.get('order')
        if order_obj is not None:
            v = self._attr_get(order_obj, ['qty','quantity','filled_quantity','quantity_filled'])
            if v is not None:
                try: return int(v)
                except: pass
        return None

    def __init__(
        self,
        id: int,
        name: str,
        symbol: str,
        starting_cash: float,
        exchange_id: int,
        wakeup_freq: str = "5s",
        size: int = 100,
        sma_window: int = 30,
        spread_floor: int = 1,           # half-spread floor (int)
        lc_use_groq: bool = True,       # keep default OFF
        lc_mode: str = "direct",         # "advice" or "direct"
        lc_every: int = 25,               # call LLM every N wakeups
        lc_timeout_s: float = 1.5,       # soft budget; skip if slower
        groq_model: str = "llama-3.1-8b-instant",
        levels: int = 10,
        subscription_freq: str = "5s",
        base_mid: Optional[int] = 100000,
        print_tools: bool = True,
        metrics_path: Optional[str] = None,
        tools_print_every: int = 5,      # throttle tool prints
        # --- bootstrap safety knobs ---
        quote_bootstrap_mode: str = "wait",  # "wait" = do not quote until inside or last trade exists; "wide" = quote wide/tiny
        bootstrap_half: int = 50,            # if mode="wide": half-spread to use when standing up
        bootstrap_qty_frac: float = 0.1,     # if mode="wide": qty = int(size * this), min 1
        **kwargs,
    ):
        # RNG normalize
        self.rng = kwargs.pop("rng", None)
        try:
            import numpy as _np
            if isinstance(self.rng, (int, _np.integer)):
                self.rng = _np.random.RandomState(int(self.rng))
            elif getattr(self.rng, "randint", None) is None and getattr(self.rng, "random_integers", None) is None:
                if getattr(self.rng, "integers", None) is not None:
                    seed = int(self.rng.integers(0, 2**31 - 1))
                    self.rng = _np.random.RandomState(seed)
                else:
                    self.rng = _np.random.RandomState()
        except Exception:
            try:
                import numpy as _np
                self.rng = _np.random.RandomState()
            except Exception:
                self.rng = None

        # absorb stray kwargs
        kwargs.pop("maker_spread_bps", None)
        kwargs.pop("lc_policy_refresh", None)
        kwargs.pop("rate_limit_rps", None)

        # fields pre-base
        self.symbol = symbol
        self.exchange_id = exchange_id
        self.levels = int(levels)
        self.subscription_freq = str(subscription_freq)
        try:
            self.subscription_freq_ns = pd.Timedelta(self.subscription_freq).value
        except Exception:
            try:
                self.subscription_freq_ns = pd.Timedelta(seconds=float(self.subscription_freq)).value
            except Exception:
                self.subscription_freq_ns = pd.Timedelta("1s").value
        self.wakeup_freq = pd.Timedelta(wakeup_freq)
        self.size = int(size)
        self.sma_window = int(sma_window)
        self.spread_floor = int(spread_floor)
        self.base_mid = int(base_mid) if base_mid is not None else 100000
        self.print_tools = bool(print_tools)
        self.tools_print_every = int(max(1, tools_print_every))
        self.metrics_path = metrics_path or os.getenv("LCMM_METRICS_DIR")

        # bootstrap quoting behavior
        self.quote_bootstrap_mode = str(quote_bootstrap_mode).lower()
        self.bootstrap_half = int(bootstrap_half)
        self.bootstrap_qty_frac = float(max(0.01, min(1.0, bootstrap_qty_frac)))


        # LLM flags
        self.lc_use_groq = bool(lc_use_groq)
        self.lc_mode = lc_mode if lc_mode in ("advice","direct") else "advice"
        self.lc_every = int(max(1, lc_every))
        self.lc_timeout_s = float(max(0.1, lc_timeout_s))

        super().__init__(
            id,
            name,
            type(self).__name__,
            self.symbol,
            starting_cash,
            self.levels,
            self.subscription_freq_ns,
            log_orders=True,
            random_state=self.rng,
        )

        # CSV sinks
        self._tools_csv_file = None
        self._tools_csv_writer = None
        self._tools_csv_path = None
        self._metrics_csv_file = None
        self._metrics_csv_writer = None
        self._metrics_csv_path = None

        # market state
        self.current_bids: List[Tuple[int, int]] = []
        self.current_asks: List[Tuple[int, int]] = []
        self.best_bid: Optional[int] = None
        self.best_ask: Optional[int] = None
        self.mid: Optional[float] = None

        # histories
        self.mid_history: deque[float] = deque(maxlen=6000)
        self.spread_history: deque[int] = deque(maxlen=6000)

        # orders registry
        self.order_meta: Dict[Any, Dict[str, Any]] = {}
        self.last_bid_int: Optional[int] = None
        self.last_ask_int: Optional[int] = None

        # fill-rate stats
        self.level_stats: Dict[str, Dict[int, Dict[str, int]]] = {
            'BUY': defaultdict(lambda: {'placed': 0, 'filled': 0}),
            'SELL': defaultdict(lambda: {'placed': 0, 'filled': 0}),
        }

        # LLM client + prompts
        self._llm_tick_counter = 0
        self._llm_last_advice: Dict[str, Any] = {"side_hint":"HOLD","half_spread_adj":0}
        self._llm_last_plan: Dict[str, Any] = {"cancel_all": False, "place": []}

        self._llm_system_prompt_advice = (
            "You are an exchange microstructure assistant for a limit-order market maker.\n"
            "You receive compact JSON tool summaries and must output ONLY a compact JSON advice object.\n"
            "Conventions: prices are integer native units. You do NOT send orders—execution is separate.\n"
            "Return JSON: {side_hint:'BUY'|'SELL'|'HOLD', half_spread_adj:int (|adj|<=3), comment:string}."
        )
        self._llm_system_prompt_direct = (
            """
        You are an AGGRESSIVE LIQUIDITY-PROVIDING market-making assistant in a limit order book.
        You must output ONLY a compact JSON action plan; the agent will execute it verbatim.

        Hard rules:
        - Always provide TWO-SIDED QUOTES
        - Prices are INTEGER native units; obey tick-size ≥ 1 and DO NOT CROSS the book (ensure SELL > BUY).
        - Prefer quoting at or 1 tick from the inside when best bid/ask are known.
        - If book info is missing, center around last_trade; if missing too, use 100000.
        - Default quantity per order is `default_qty` unless a clear risk reason to reduce.
        - Keep quotes within ±3*tick of `mid` when uncertain.
        - Output strictly valid JSON; no prose, no code fences, no extra fields.
        - If new quotes differ from last cycle, set "cancel_all": true (replace-style quoting).
        - If there is no clear signal, ALWAYS return {"side_hint":"HOLD", "half_spread_adj":0}.
        - If there is NO inside (best_bid/best_ask) AND NO last_trade, ALWAYS return HOLD with half_spread_adj: 0.

        Return JSON exactly:
        {"cancel_all": bool, "place": [
        {"side":"BUY","price":int,"qty":int},
        {"side":"SELL","price":int,"qty":int}
        ]}
        - Always include at least these two orders (one BUY, one SELL). At most 2 orders total.
        - Use integer prices; ensure SELL price > BUY price after rounding.
            """
        )

        self.lc = None
        if self.lc_use_groq and ChatGroq is not None:
            try:
                # keep generation short to reduce latency
                self.lc = ChatGroq(model=groq_model, temperature=0, max_tokens=128)
                print(f"[LCMM] Groq LLM initialized ({self.lc_mode} mode, every {self.lc_every} wakeups)")
            except Exception as e:
                print(f"[LCMM] Groq init failed: {e}")

    # ----- lifecycle -----
    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        # CSVs
        try:
            metrics_dir = (
                self.metrics_path
                or getattr(self, 'log_dir', None)
                or getattr(self, 'output_dir', None)
                or '.'
            )
            os.makedirs(metrics_dir, exist_ok=True)
            print(f"[LCMM][TOOLS CSV] dir={metrics_dir}")
            # tools
            self._tools_csv_path = os.path.join(metrics_dir, f"{self.name}_tools.csv")
            print(f"[LCMM][TOOLS CSV] file={self._tools_csv_path}")
            self._tools_csv_file = open(self._tools_csv_path, 'a', newline='')
            fieldnames = ['t','cash','pos','mid','exposure','open_buy','open_sell',
                          'top_buy_levels','top_sell_levels','vol','fill_rates']
            self._tools_csv_writer = csv.DictWriter(self._tools_csv_file, fieldnames=fieldnames)
            if self._tools_csv_file.tell() == 0:
                self._tools_csv_writer.writeheader(); self._tools_csv_file.flush()
            # metrics mirror
            self._metrics_csv_path = os.path.join(metrics_dir, f"{self.name}_metrics.csv")
            print(f"[LCMM][METRICS CSV] file={self._metrics_csv_path}")
            self._metrics_csv_file = open(self._metrics_csv_path, 'a', newline='')
            self._metrics_csv_writer = csv.DictWriter(self._metrics_csv_file, fieldnames=fieldnames)
            if self._metrics_csv_file.tell() == 0:
                self._metrics_csv_writer.writeheader(); self._metrics_csv_file.flush()
        except Exception as e:
            print(f"[LCMM][TOOLS CSV WARN] could not init csv: {e}")

        self._subscribe_market_data()
        # first wake
        self.setWakeup(startTime + pd.Timedelta(milliseconds=int(np.random.randint(0,1500))))
        print(f"[LCMM] start: symbol={self.symbol} wakeup_freq={self.wakeup_freq}")

    def kernelStopping(self):
        # close CSVs
        for _fh in ("_tools_csv_file","_metrics_csv_file"):
            try:
                f = getattr(self, _fh, None)
                if f:
                    try: f.flush()
                    except: pass
                    try: f.close()
                    except: pass
            except: pass
        # backfill mark
        try:
            if not hasattr(self, "last_trade") or self.last_trade is None:
                self.last_trade = {}
            if self.symbol not in self.last_trade or not self.last_trade[self.symbol]:
                fallback = self.mid if (self.mid is not None) else float(self.base_mid)
                self.last_trade[self.symbol] = float(fallback)
                print(f"[LCMM][STOP] backfilled last_trade[{self.symbol}]={self.last_trade[self.symbol]}")
        except Exception as e:
            print(f"[LCMM][STOP WARN] {e}")
        super().kernelStopping()

    def getWakeFrequency(self):
        return self.wakeup_freq

    # ----- subscription -----
    def _subscribe_market_data(self):
        sym = self.symbol
        tried = []
        def _try(name, *args):
            if hasattr(self, name):
                try:
                    getattr(self, name)(*args); tried.append(f"{name}{args}"); return True
                except Exception: return False
            return False
        if _try("requestDataSubscription", sym, self.levels, self.subscription_freq_ns): pass
        elif _try("requestDataSubscription", sym): pass
        elif _try("subscribeLevel2", sym): pass
        elif _try("subscribeBook", sym): pass
        elif _try("subscribe_market_data", sym): pass
        elif _try("subscribeMarketData", sym): pass
        elif _try("subscribe", sym): pass
        else:
            print(f"[LCMM][WARN] no known subscription method found for {sym}"); return
        print(f"[LCMM] subscribed via {tried[0]} for {sym}")

    # ----- main loop -----
    def wakeup(self, currentTime):
        super().wakeup(currentTime)

        mid = self._compute_mid()
        half = self._suggest_spread()
        q = int(self.size)
        tick = self._tick()
                # --- bootstrap safety: do not quote into a vacuum ---
        have_inside = (self.best_bid is not None and self.best_ask is not None and self.best_ask > self.best_bid)
        have_trade = False
        try:
            if hasattr(self, "last_trade") and isinstance(self.last_trade, dict):
                have_trade = (self.last_trade.get(self.symbol) is not None)
        except Exception:
            pass

        if not have_inside and not have_trade:
            if self.quote_bootstrap_mode == "wait":
                print(f"[LCMM][BOOTSTRAP] no inside & no last trade yet → standing down")
                self.setWakeup(currentTime + self.wakeup_freq)
                return
            elif self.quote_bootstrap_mode == "wide":
                # fall back to very wide, tiny-size quotes around base_mid
                mid = float(self.base_mid)
                half = max(self.bootstrap_half, self.spread_floor, 1)
                q = max(1, int(round(self.size * self.bootstrap_qty_frac)))
                print(f"[LCMM][BOOTSTRAP] wide mode: mid={int(mid)} half={half} qty={q}")
        if mid is None: mid = float(self.base_mid)

        # baseline mean-reversion hint
        side_hint = None
        if len(self.mid_history) >= self.sma_window:
            sma = float(np.mean(list(self.mid_history)[-self.sma_window:]))
            diff = (mid - sma)
            if diff > tick: side_hint = "SELL"
            elif diff < -tick: side_hint = "BUY"

        # ---------- LLM throttle + mode ----------
        self._llm_tick_counter += 1
        if self.lc_use_groq and (self.lc is not None) and (self._llm_tick_counter % self.lc_every == 0):
            inv = self.tool_inventory()
            oo  = self.tool_open_orders_summary()
            fr  = self.tool_fill_rates_summary(max_levels=5)
            rv  = self.tool_rolling_vol(window=120)

            try:
                t0 = time.time()
                if self.lc_mode == "advice":
                    self._llm_last_advice = self._llm_advice(currentTime, mid, half, tick, inv, oo, fr, rv)
                    # blend advice into baseline
                    half = max(self.spread_floor, max(1, int(half + self._llm_last_advice.get("half_spread_adj", 0))))
                    sh = self._llm_last_advice.get("side_hint")
                    if sh in ("BUY","SELL"): side_hint = sh
                else:
                    self._llm_last_plan = self._llm_direct_plan(currentTime, mid, half, tick, inv, oo, fr, rv)
                dt = time.time() - t0
                if dt > self.lc_timeout_s:
                    print(f"[LCMM][LLM WARN] call exceeded {self.lc_timeout_s:.2f}s; consider raising lc_every")
            except Exception as e:
                print(f"[LCMM][LLM WARN] {e}")

        # ---------- Execution ----------
        if self.lc_use_groq and self.lc_mode == "direct" and self._llm_last_plan:
            self._execute_direct_plan(mid=mid, tick=tick, plan=self._llm_last_plan)
        else:
            # symmetric quotes around mid (advice/baseline)
            bid = int(round(mid - half)); ask = int(round(mid + half))
            if bid >= ask: ask = bid + tick
            if side_hint == "BUY":  bid = min(bid + tick, int(round(mid)))
            if side_hint == "SELL": ask = max(ask - tick, int(round(mid)) + tick)
            if (bid != self.last_bid_int) or (ask != self.last_ask_int):
                self._cancel_all()
                if bid < ask and q > 0:
                    self._place_limit("BUY", bid, q)
                    self._place_limit("SELL", ask, q)
                    print(f"[LCMM][QUOTE] t={currentTime} bid={bid} ask={ask} qty={q}")
                self.last_bid_int, self.last_ask_int = bid, ask

        # ---------- tools (throttled prints, always CSV) ----------
        inv = self.tool_inventory(); oo = self.tool_open_orders_summary()
        fr = self.tool_fill_rates_summary(max_levels=5); rv = self.tool_rolling_vol(window=120)
        if self.print_tools and (self._llm_tick_counter % self.tools_print_every == 0):
            print(f"[LCMM][TOOLS] inv={inv} open_orders={oo} vol={rv:.6f} (logret std)")
            if fr: print(f"[LCMM][TOOLS] fill_rates={fr}")
        try:
            if self._tools_csv_writer is not None:
                row = {'t': str(currentTime),'cash': inv.get('cash'),'pos': inv.get('pos'),'mid': inv.get('mid'),
                       'exposure': inv.get('exposure'),'open_buy': oo.get('open_buy'),'open_sell': oo.get('open_sell'),
                       'top_buy_levels': json.dumps(oo.get('top_buy_levels'), separators=(',',':')),
                       'top_sell_levels': json.dumps(oo.get('top_sell_levels'), separators=(',',':')),
                       'vol': rv,'fill_rates': json.dumps(fr, separators=(',',':'))}
                self._tools_csv_writer.writerow(row); self._tools_csv_file.flush()
            if self._metrics_csv_writer is not None:
                self._metrics_csv_writer.writerow(row); self._metrics_csv_file.flush()
        except Exception as e:
            print(f"[LCMM][TOOLS CSV WARN] write failed: {e}")

        self.setWakeup(currentTime + self.wakeup_freq)

    # ----- receive messages (unchanged structure, trimmed) -----
    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        body = getattr(msg, "body", None)
        if not isinstance(body, dict): return
        msg_type = (body.get("msg") or body.get("type") or "").upper()

        if msg_type == "MARKET_DATA" and not hasattr(self, "_printed_first_md"):
            try: print(f"[LCMM][MD] keys={list(body.keys())}")
            except: pass
            self._printed_first_md = True

        if msg_type in ("ORDER_ACCEPTED","ORDER_ACK","ACK"):
            oid = self._extract_order_id(body)
            side = self._extract_side_any(body, oid)
            px   = self._extract_price_any(body)
            qty  = self._extract_qty_any(body)
            if oid is not None:
                level = None
                if self.best_bid is not None and self.best_ask is not None and side and px is not None:
                    try:
                        pxi = int(px); tick = self._tick()
                        level = max(0,(self.best_bid - pxi)//max(tick,1)) if side.upper().startswith('B') else max(0,(pxi - self.best_ask)//max(tick,1))
                    except: level = None
                self.order_meta[oid] = {'side': side,'price': int(px) if px is not None else None,
                                        'qty_open': int(qty) if qty is not None else None,'ts': currentTime,
                                        'level_ticks': level}
                if side and qty is not None:
                    lv = int(level) if (level is not None) else -1
                    self.level_stats[side.split()[0] if ' ' in side else side][lv]['placed'] += int(qty)
            print(f"[LCMM][ACK] t={currentTime} oid={oid}")

        elif msg_type in ("ORDER_REJECTED","REJECTED","REJECT","ORDER_CANCELLED","CANCELLED"):
            oid = self._extract_order_id(body)
            print(f"[LCMM][REJECT/CANCEL] t={currentTime} oid={oid} reason={body.get('reason')}")
            if oid in self.order_meta:
                try: self.order_meta[oid]["qty_open"] = 0
                except: pass

        if (msg_type in ("ORDER_EXECUTED","ORDER_FILLED","EXECUTION","FILL")) or any(
            k in body for k in ("fill_price","execution_price","avg_price","fill_qty","filled_quantity","quantity_filled","execution_qty")
        ):
            oid = self._extract_order_id(body)
            side = self._extract_side_any(body, oid)
            price = self._extract_price_any(body)
            qty = self._extract_qty_any(body)
            if (qty is not None) and (price is not None) and side:
                print(f"[LCMM][FILL] t={currentTime} side={side} qty={int(qty)} px={int(price)} oid={oid}")
            if oid in self.order_meta:
                try: self.order_meta[oid]['qty_open'] = max(0, int(self.order_meta[oid].get('qty_open') or 0) - int(qty or 0))
                except: pass
                lv = self.order_meta[oid].get('level_ticks'); lv = int(lv) if (lv is not None) else -1
                if side: self.level_stats[side.split()[0] if ' ' in side else side][lv]['filled'] += int(qty or 0)
            else:
                if side: self.level_stats[side.split()[0] if ' ' in side else side][-1]['filled'] += int(qty or 0)

        if msg_type in ("TRADE","LAST_TRADE") and (body.get("price") is not None or body.get("trade_price") is not None):
            px = body.get("price") or body.get("trade_price")
            try:
                self.mid = float(px); self.mid_history.append(self.mid)
                if not hasattr(self, "last_trade") or self.last_trade is None: self.last_trade = {}
                self.last_trade[self.symbol] = float(px)
            except: pass

        book = body.get("book") or body.get("lob")
        bids = asks = None
        if isinstance(book, dict):
            bids = book.get("bids") or book.get("bid"); asks = book.get("asks") or book.get("ask")
        if bids is None and 'bids' in body and 'asks' in body:
            bids = body.get('bids'); asks = body.get('asks')
        if bids is not None or asks is not None:
            def _pair(x):
                try: return (int(x[0]), int(x[1]))
                except: return None
            self.current_bids = [p for p in (_pair(x) for x in (bids or [])) if p]
            self.current_asks = [p for p in (_pair(x) for x in (asks or [])) if p]
            bb = self.current_bids[0][0] if self.current_bids else None
            ba = self.current_asks[0][0] if self.current_asks else None
            if bb is not None and ba is not None and ba > bb:
                self.best_bid, self.best_ask = bb, ba
                self.mid = 0.5 * (bb + ba)
                self.mid_history.append(self.mid)
                self.spread_history.append(ba - bb)
                if not hasattr(self, "_printed_first_book"):
                    print(f"[LCMM][BOOK] L1 bid/ask = {self.best_bid} / {self.best_ask}; bids[0:2]={self.current_bids[:2]} asks[0:2]={self.current_asks[:2]}")
                    self._printed_first_book = True

    # ----- tools -----
    def tool_inventory(self) -> Dict[str, Any]:
        cash = None; pos = None
        try: cash = float(self.holdings.get('CASH'))
        except: pass
        try:
            if hasattr(self, 'getPosition'): pos = int(self.getPosition(self.symbol))
            else: pos = int(self.holdings.get(self.symbol, 0))
        except: pos = None
        mid = float(self.mid) if self.mid is not None else float(self.base_mid)
        exposure = None if (pos is None) else pos * mid
        return {'cash': cash, 'pos': pos, 'mid': mid, 'exposure': exposure}

    def tool_open_orders_summary(self) -> Dict[str, Any]:
        buys = [o for o in self.order_meta.values() if (o.get('side') or '').upper().startswith('B') and (o.get('qty_open') or 0) > 0]
        sells= [o for o in self.order_meta.values() if (o.get('side') or '').upper().startswith('S') and (o.get('qty_open') or 0) > 0]
        def _levels(lst):
            lv = [o.get('level_ticks') for o in lst if o.get('level_ticks') is not None]
            if not lv: return []
            vals, counts = np.unique(np.array(lv, dtype=int), return_counts=True)
            order = np.argsort(-counts)
            top = [(int(vals[i]), int(counts[i])) for i in order[:2]]
            return top
        return {'open_buy': len(buys),'open_sell': len(sells),'top_buy_levels': _levels(buys),'top_sell_levels': _levels(sells)}

    def tool_fill_rates_summary(self, max_levels: int = 5) -> Dict[str, Dict[int, float]]:
        out: Dict[str, Dict[int, float]] = {'BUY': {}, 'SELL': {}}
        for side in ('BUY','SELL'):
            placed_map: Dict[int, int] = defaultdict(int)
            filled_map: Dict[int, int] = defaultdict(int)
            for L, stats in self.level_stats[side].items():
                Lint = int(L); bucket = Lint if (-1 <= Lint <= max_levels) else (max_levels + 1)
                placed_map[bucket] += int(stats.get('placed', 0)); filled_map[bucket] += int(stats.get('filled', 0))
            for L in range(-1, max_levels + 1):
                p = placed_map.get(L, 0); f = filled_map.get(L, 0)
                out[side][L] = (f / p) if p > 0 else 0.0
            p = placed_map.get(max_levels + 1, 0); f = filled_map.get(max_levels + 1, 0)
            out[side][max_levels + 1] = (f / p) if p > 0 else 0.0
        return out

    def tool_rolling_vol(self, window: int = 120) -> float:
        if len(self.mid_history) < max(window, 5): return 0.0
        arr = np.array(list(self.mid_history)[-window:], dtype=float); arr = arr[arr > 0]
        if len(arr) < 5: return 0.0
        rets = np.diff(np.log(arr))
        return float(np.std(rets))

    # ----- LLM: advice (side/spread) -----
    def _llm_advice(self, currentTime, mid, half, tick, inv, open_orders, fill_rates, vol) -> dict:
        if self.lc is None:
            return {"side_hint":"HOLD","half_spread_adj":0,"comment":"LLM disabled"}
        context = {
            "now": str(currentTime), "symbol": self.symbol, "tick": int(tick),
            "mid": int(round(mid)) if mid is not None else None,
            "current_half": int(half), "inventory": inv, "open_orders": open_orders,
            "fill_rates": fill_rates, "rolling_vol": float(vol),
            "have_inside": bool(have_inside),
            "have_trade": bool(have_trade)

        }
        user_msg = "TOOLS(JSON):" + json.dumps(context, separators=(',',':')) + \
                   "\nReturn ONLY JSON: {side_hint:'BUY'|'SELL'|'HOLD', half_spread_adj:int, comment:string}"
        try:
            start = time.time()
            resp = self.lc.invoke(self._llm_system_prompt_advice + "\n" + user_msg)
            if time.time() - start > self.lc_timeout_s: raise TimeoutError("Groq slow")
            txt = getattr(resp, 'content', None) or str(resp)
            data = json.loads(txt)
        except Exception:
            data = {"side_hint":"HOLD","half_spread_adj":0,"comment":"unparsable"}
        side = str(data.get("side_hint","HOLD")).upper()
        if side not in ("BUY","SELL","HOLD"): side = "HOLD"
        try: adj = int(data.get("half_spread_adj",0))
        except: adj = 0
        adj = max(-3, min(3, adj))
        return {"side_hint": side, "half_spread_adj": adj, "comment": data.get("comment","")}

    # ----- LLM: direct orders -----
    def _llm_direct_plan(self, currentTime, mid, half, tick, inv, open_orders, fill_rates, vol) -> dict:
        """
        Direct plan: ALWAYS return two-sided quotes (one BUY, one SELL).
        Context includes best bid/ask, last_trade, base_mid, and default_qty for sizing.
        """
        # local helpers for fallbacks
        def _center_price():
            # prefer computed mid; else last_trade; else base_mid; else hardcoded 100000
            m = int(round(mid)) if mid is not None else None
            if m is not None:
                return m
            try:
                if hasattr(self, 'last_trade') and isinstance(self.last_trade, dict):
                    lt = self.last_trade.get(self.symbol)
                    if lt is not None:
                        return int(round(lt))
            except Exception:
                pass
            return int(self.base_mid) if self.base_mid is not None else 100000

        if self.lc is None:
            c = _center_price()
            return {"cancel_all": False, "place": [
                {"side": "BUY",  "price": c - int(tick), "qty": int(self.size)},
                {"side": "SELL", "price": c + int(tick), "qty": int(self.size)}
            ]}

        # gather extra context for LLM
        last_trade_val = None
        try:
            if hasattr(self, 'last_trade') and isinstance(self.last_trade, dict):
                last_trade_val = self.last_trade.get(self.symbol)
        except Exception:
            last_trade_val = None

        context = {
            "now": str(currentTime),
            "symbol": self.symbol,
            "tick": int(tick),
            "mid": int(round(mid)) if mid is not None else None,
            "best_bid": int(self.best_bid) if self.best_bid is not None else None,
            "best_ask": int(self.best_ask) if self.best_ask is not None else None,
            "last_trade": int(round(last_trade_val)) if last_trade_val is not None else None,
            "base_mid": int(self.base_mid),
            "current_half": int(half),
            "default_qty": int(self.size),
            "inventory": inv,
            "open_orders": open_orders,
            "fill_rates": fill_rates,
            "rolling_vol": float(vol),
        }

        user_msg = (
            "TOOLS(JSON):" + json.dumps(context, separators=(',',':')) +
            "\nCompose two-sided quotes AGGRESSIVELY but safely:\n"
            "- If best_bid/best_ask known: BUY at max(best_bid, mid-1*tick), SELL at min(best_ask, mid+1*tick).\n"
            "- If unknown: center = last_trade or 100000; set BUY=center-1*tick, SELL=center+1*tick.\n"
            "- Clamp within ±3*tick of mid if provided. Ensure integers and SELL>BUY. Use default_qty unless risk reduction is obvious.\n"
            "Return ONLY the JSON action plan specified."
        )

        # call Groq and parse
        try:
            start = time.time()
            resp = self.lc.invoke(self._llm_system_prompt_direct + "\n" + user_msg)
            if time.time() - start > self.lc_timeout_s:
                raise TimeoutError("Groq slow")
            txt = getattr(resp, 'content', None) or str(resp)
            plan = json.loads(txt)
        except Exception:
            # fallback: symmetric around center
            c = _center_price()
            plan = {"cancel_all": False, "place": [
                {"side": "BUY",  "price": c - int(tick), "qty": int(self.size)},
                {"side": "SELL", "price": c + int(tick), "qty": int(self.size)}
            ]}

        # sanitize & enforce two-sided, max 2 orders
        place = []
        for o in (plan.get("place") or [])[:2]:
            try:
                side = str(o.get("side","")).upper()
                if side not in ("BUY","SELL"):
                    continue
                px = int(o.get("price"))
                qty = int(o.get("qty")) if int(o.get("qty", 0)) > 0 else int(self.size)

                # anchor around mid/center, keep within ±3*tick
                m = int(round(mid)) if mid is not None else _center_price()
                px = int(round(px))
                if side == "BUY":
                    # cap overly aggressive buys above mid+3*tick
                    px = min(px, m + 3*int(tick))
                else:
                    # cap overly aggressive sells below mid-3*tick
                    px = max(px, m - 3*int(tick))
                place.append({"side": side, "price": px, "qty": qty})
            except Exception:
                continue

        # ensure at least one on each side
        sides = {p["side"] for p in place}
        if "BUY" not in sides or "SELL" not in sides:
            c = _center_price()
            place = [
                {"side": "BUY",  "price": c - int(tick), "qty": int(self.size)},
                {"side": "SELL", "price": c + int(tick), "qty": int(self.size)}
            ]

        # enforce SELL > BUY
        try:
            bmax = max(p["price"] for p in place if p["side"] == "BUY")
            smin = min(p["price"] for p in place if p["side"] == "SELL")
            if not (smin > bmax):
                c = _center_price()
                place = [
                    {"side": "BUY",  "price": c - int(tick), "qty": int(self.size)},
                    {"side": "SELL", "price": c + int(tick), "qty": int(self.size)}
                ]
        except Exception:
            pass

        return {"cancel_all": bool(plan.get("cancel_all", False)), "place": place}

    # ----- Execute a direct plan -----
    def _execute_direct_plan(self, *, mid: float, tick: int, plan: dict):
        try:
            if plan.get("cancel_all", False):
                self._cancel_all()
            # place
            for o in plan.get("place", []):
                self._place_limit(o["side"], o["price"], o["qty"])
            # remember last quotes (best effort) to avoid auto re-quote
            buys = [o for o in plan.get("place", []) if o["side"] == "BUY"]
            sells= [o for o in plan.get("place", []) if o["side"] == "SELL"]
            if buys:  self.last_bid_int = max([int(b["price"]) for b in buys])
            if sells: self.last_ask_int = min([int(a["price"]) for a in sells])
            print(f"[LCMM][PLAN] cancel_all={plan.get('cancel_all', False)} place={plan.get('place', [])}")
        except Exception as e:
            print(f"[LCMM][PLAN WARN] {e}")

    # ----- helpers -----
    def _compute_mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None and self.best_ask > self.best_bid:
            self.mid = 0.5 * (self.best_bid + self.best_ask)
        return self.mid

    def _suggest_spread(self) -> int:
        if self.best_bid is not None and self.best_ask is not None and self.best_ask > self.best_bid:
            inside = int(self.best_ask - self.best_bid)
            # keep at least 1* tick and avoid over-tightening at boot
            half = max(self.spread_floor, max(1, int(math.ceil(0.8 * inside))))
            return half

        if not self.spread_history:
            return max(self._tick(), self.spread_floor, 1)
        arr = np.array(list(self.spread_history)[-60:], dtype=float); arr = arr[arr > 0]
        if arr.size == 0: return max(self._tick(), self.spread_floor, 1)
        half = int(round(0.5 * float(np.median(arr))))
        return max(half, self.spread_floor, 1)

    def _tick(self) -> int:
        if self.spread_history:
            arr = np.array(list(self.spread_history)[-60:], dtype=float); arr = arr[arr > 0]
            if arr.size: return max(int(round(np.percentile(arr, 10) / 10.0)), 1)
        return 1

    # ----- orders -----
    def _place_limit(self, side: str, price: int | float, qty: int):
        """Place a limit order robustly across ABIDES forks.
        - Prefer keyword arguments by name (is_buy, quantity/qty, limit_price/price).
        - If positional-only, detect signature order and call accordingly.
        - Update local registries and stats.
        """
        is_buy_flag = str(side).upper().startswith("B")
        sym = self.symbol
        try:
            px_int = int(round(price))
        except Exception:
            px_int = int(price)
        q = int(qty)

        # Snapshot base class order map keys before placing (for id diff)
        try:
            base_before = set(self.orders.keys()) if isinstance(getattr(self, "orders", None), dict) else set()
        except Exception:
            base_before = set()

        # Resolve API handle
        fn = None
        if hasattr(self, "place_limit_order"):
            fn = getattr(self, "place_limit_order")
        elif hasattr(self, "placeLimitOrder"):
            fn = getattr(self, "placeLimitOrder")
        else:
            print("[LCMM][WARN] no limit-order API found on this fork")
            return None

        import inspect
        real_oid = None
        last_err = None

        # 1) Try keyword call by parameter names (best)
        try:
            params = list(inspect.signature(fn).parameters.keys())
            kwargs = {}
            # common param name variants
            if "symbol" in params:
                kwargs["symbol"] = sym
            if "sym" in params and "symbol" not in kwargs:
                kwargs["sym"] = sym

            # quantity
            if "quantity" in params:
                kwargs["quantity"] = q
            elif "qty" in params:
                kwargs["qty"] = q

            # is_buy
            if "is_buy" in params:
                kwargs["is_buy"] = bool(is_buy_flag)
            elif "buy" in params:
                kwargs["buy"] = bool(is_buy_flag)

            # price
            if "limit_price" in params:
                kwargs["limit_price"] = px_int
            elif "price" in params:
                kwargs["price"] = px_int
            elif "px" in params:
                kwargs["px"] = px_int

            # Only do the keyword call if we actually mapped all roles
            if ("symbol" in kwargs or "sym" in kwargs) and \
            any(k in kwargs for k in ("quantity","qty")) and \
            any(k in kwargs for k in ("is_buy","buy")) and \
            any(k in kwargs for k in ("limit_price","price","px")):
                oid = fn(**kwargs)
            else:
                raise TypeError("Keyword-arg mapping incomplete; falling back to positional detection.")
        except Exception as e_kw:
            last_err = e_kw
            oid = None

            # 2) Detect positional signature order safely
            try:
                # common variants:
                # A) (symbol, is_buy, qty, price)
                # B) (symbol, qty, is_buy, price)
                # C) (symbol, is_buy, qty, limit_price)
                # D) (symbol, qty, is_buy, limit_price)
                tried = []

                # Try A/C first only if the function’s second param looks like 'is_buy'
                sig = inspect.signature(fn)
                names = [p.name for p in sig.parameters.values()]
                call_ok = False

                # Helper to attempt and verify
                def _attempt(args):
                    nonlocal real_oid, call_ok
                    o = fn(*args)
                    # If it returned an object, we try to inspect; else rely on ACK later
                    real_oid_local = None
                    if isinstance(o, (int, str)):
                        real_oid_local = o
                    else:
                        # try extracting id from object
                        for attr in ("order_id", "id", "agent_order_id", "agentOrderId"):
                            try:
                                v = getattr(o, attr)
                                if v is not None:
                                    real_oid_local = v
                                    break
                            except Exception:
                                pass
                        if real_oid_local is None:
                            try:
                                real_oid_local = o.__dict__.get("order_id")
                            except Exception:
                                pass
                    real_oid = real_oid_local
                    call_ok = True
                    return o

                # Try (symbol, qty, is_buy, price)
                try:
                    tried.append("positional: (sym, qty, is_buy, price)")
                    _attempt([sym, q, bool(is_buy_flag), px_int])
                except Exception:
                    # Try (symbol, is_buy, qty, price)
                    try:
                        tried.append("positional: (sym, is_buy, qty, price)")
                        _attempt([sym, bool(is_buy_flag), q, px_int])
                    except Exception:
                        # Try limit_price naming differences—at positional level it’s same int
                        pass

                if not call_ok:
                    raise TypeError(f"Could not match positional signature; tried {tried}")
            except Exception as e_pos:
                last_err = e_pos

        # Attempt to derive id by diff if we still don't have one
        if real_oid is None:
            try:
                base_after = set(self.orders.keys()) if isinstance(getattr(self, "orders", None), dict) else set()
                new_keys = list(base_after - base_before)
                if new_keys:
                    real_oid = new_keys[0]
            except Exception:
                pass

        if real_oid is None and last_err:
            print(f"[LCMM][PLACE WARN] could not derive order_id: {last_err}")
        elif real_oid is None:
            # Harmless; ACK will arrive later
            pass

        # Update local registry with the intended side/qty/price immediately
        if real_oid is not None:
            level = None
            if self.best_bid is not None and self.best_ask is not None:
                tick = self._tick()
                level = max(0, (self.best_bid - px_int)//max(tick,1)) if is_buy_flag else max(0, (px_int - self.best_ask)//max(tick,1))
            self.order_meta[real_oid] = {
                "side": "BUY" if is_buy_flag else "SELL",
                "price": px_int,
                "qty_open": q,
                "ts": None,
                "level_ticks": level,
            }
            lv = int(level) if (level is not None) else -1
            self.level_stats["BUY" if is_buy_flag else "SELL"][lv]["placed"] += q

        return real_oid
        def _extract_from_obj(x):
            for attr in ("order_id","id","agent_order_id","agentOrderId"):
                try:
                    v = getattr(x, attr)
                    if v is not None: return v
                except: pass
            try: return x.__dict__.get('order_id')
            except: return None

        real_oid = None
        if isinstance(oid, (int, str)): real_oid = oid
        if real_oid is None and oid is not None: real_oid = _extract_from_obj(oid)
        if real_oid is None:
            try:
                base_after = set(self.orders.keys()) if isinstance(self.orders, dict) else set()
                new_keys = list(base_after - base_before)
                if new_keys: real_oid = new_keys[0]
            except: pass

        if real_oid is None and last_type_err:
            print(f"[LCMM][PLACE WARN] signature mismatch fallback used; could not derive order_id: {last_type_err}")
        elif real_oid is None:
            #print("[LCMM][PLACE WARN] placed order but could not derive order_id (no ACK yet?)")
            pass

        if real_oid is not None:
            level = None
            if self.best_bid is not None and self.best_ask is not None:
                tick = self._tick()
                level = max(0, (self.best_bid - px_int)//max(tick,1)) if is_buy else max(0, (px_int - self.best_ask)//max(tick,1))
            self.order_meta[real_oid] = {'side':'BUY' if is_buy else 'SELL','price':px_int,'qty_open':q,'ts':None,'level_ticks':level}
            lv = int(level) if (level is not None) else -1
            self.level_stats['BUY' if is_buy else 'SELL'][lv]['placed'] += q
        return real_oid

    def _cancel_all(self):
        sym = self.symbol
        out = None
        try:
            if hasattr(self, "cancel_all_orders"):
                out = self.cancel_all_orders(sym)
            elif hasattr(self, "cancelAllOrders"):
                out = self.cancelAllOrders(sym)
        except Exception as e:
            print(f"[LCMM][CANCEL-ALL WARN] {e}")
            out = None

        # If batch cancel isn’t supported, best-effort per-oid
        if out is None and getattr(self, "resting_ids", None):
            for oid in list(self.resting_ids):
                try:
                    if hasattr(self, "cancelOrder"):
                        self.cancelOrder(oid)
                    elif hasattr(self, "cancel_order"):
                        self.cancel_order(oid)
                except Exception as e:
                    print(f"[LCMM][CANCEL WARN] oid={oid} err={e}")
            self.resting_ids.clear()

        # --- local close-out so tools don't over-count stale opens ---
        try:
            for oid, meta in self.order_meta.items():
                if meta and int(meta.get("qty_open") or 0) > 0:
                    meta["qty_open"] = 0
        except Exception:
            pass


        # Best-effort per-oid cancel if the fork doesn't support the batch call
        if out is None and getattr(self, "resting_ids", None):
            for oid in list(self.resting_ids):
                try:
                    if hasattr(self, "cancelOrder"):
                        self.cancelOrder(oid)
                    elif hasattr(self, "cancel_order"):
                        self.cancel_order(oid)
                except Exception as e:
                    print(f"[LCMM][CANCEL WARN] oid={oid} err={e}")
            self.resting_ids.clear()

        # --- local close-out so tools don't over-count stale opens ---
        try:
            for oid, meta in self.order_meta.items():
                if meta and int(meta.get("qty_open") or 0) > 0:
                    meta["qty_open"] = 0
        except Exception:
            pass


    # ----- utils -----
    @staticmethod
    def _extract_order_id(body: dict):
        oid = body.get("order_id")
        if oid is not None: return oid
        order = body.get("order")
        if isinstance(order, dict):
            return order.get("order_id") or order.get("id")
        for attr in ("order_id","id","agent_order_id","agentOrderId"):
            try: return getattr(order, attr)
            except Exception: pass
        return None
