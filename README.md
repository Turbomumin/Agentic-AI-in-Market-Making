# Agentic-AI-in-Market-Making
The key question is what strategies and measures an agent will take when in a liquidity providing role. A multi-staged approach will be taken, where additional variables are introduced continuonsly. Agents are to be designed to avoid specifically look-ahead bias, survivorship bias, and overfitting bias. Therefore, the minimal amount of inputs to model design, while avoiding these three biases, will be a core design philosophy.

In their complete state, agents will be able to place orders at multiple levels of the order book. The simulated order book will be limited to five ticks around L1 prices. Agents will be limited to trading at discrete time intervals to perserve simplicity and computing power. The amount of trading opportunities is determined as the minimal amount of trading needed to give opportunities for longer term strategies.

This experiment will use a multi-hypothesis framework, with individual assumptions being tested with modifying agent parameters

## Agent reasoning and decision making
For agent modelling, LangChain will be used. All agents will keep past reasoning in their memory to be able to maintain strategies. Only the finalized trades will be considered public information, so agents will not able to access the current or past reasoning of other agents, only the actual trades that have taken place. Agents will consider current inventory levels and risk assessment. 
Agents will keep track of unfulfilled orders, and edit those at each time interval. If the order meets a counterparty's order, it will be considered filled, removied from the order book, and added to the agents inventory. Agents will see the last simulated order book, and make trades based on those. The finalized order book for a given time will be net of all assets purchased and sold.
*Market Maker:* Agents will be given an initial prompt and training data. The market making agent will be asked to add orders to the order book by selling securities from its current inventory, and remove orders by buying assets from other agents, which is added to its current inventory.
*Traders:* Trading agents are given an initial prompt, which outlines their specific trading methodology. No training data will be given to these agents. These agents will make their decisions at the same time as the market maker.

### Agents (subject to change)
Prelimiarly, the market environment will be comprised of the market making agent, three investors, and a world agent. The world agent will make all trades which are not done by the four other agents. Therefore, five agents will make trades simultaneously. Each agent will have a predefined buying power, measured in dollars.

*Noise Trader:* One agent will be a noise trader, which makes trades on pseudo-random basis with no strucutred long-term strategy in mind. The intuition behind this agent is that as retail investor trades as grouped, their cohesive trades will be essentially random. These agents will make trades primarily at the market price, with deviations only if needed to fulfill the order at the next time stage.
*Momentum Trader:* The momentum trader will use a momentum based strategy, where technical indicators are used to make decisions on bid and ask order placement. Technical analysis has become more popular in markets, and their incorporation is required to 
*3rd agent:* To be expanded

## Testing hypotheses (subject to change)
*1: Can the MM agent form structured startegies with acceptable performance in a market with only the MM agent and the world agent?
*2: Can the MM agent maintain performance and market liqudity when additional demand is introduced, i.e. the other trading agents are incorporated.
*3: How will the MM agent's decision making change as experiment duration is extended?
*4: How will the MM agent's decision making change when its nominal starting amount is reduced?
*5: How will the MM agent react to an extreme market liquidity event?
*6: 
