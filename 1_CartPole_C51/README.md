# C51
본 markdown 파일은 2017년에 발표된 논문 [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) 의 내용에 대해 설명하는 논문입니다.

<img src="./Images/paper.png" alt="paper" class = "center" style="width: 700px;"/>

 

 일반적인 강화학습은 task를 다양하게 시도해보고 평균 reward를 예측합니다. 그리고 이 예측을 통해 어떻게 행동할지를 결정합니다. 하지만!! 환경이 랜덤성을 포함하고 있으면 같은 상황에서도 다음 state에서 받는 reward가 변할 수 있습니다. 이런 경우 다음 state에서 받을 수 있는 reward를 확률적으로 나타내면 다음과 같이 multi-modal한 분포로 표현할 수 있습니다. 

<img src="./Images/bimodal_distribution.png" alt="bimodal" class = "center" style="width: 300px;"/>

 강화학습을 주로 테스트하는 게임들은 랜덤성을 가지고 있지 않은 경우가 많지만 실제 환경은 대부분의 경우 랜덤성을 포함하고 있습니다. 이에 따라 랜덤성을 가진 환경에서 더 정확한 예측을 하는 알고리즘이 필요합니다.

 일반적인 강화학습 알고리즘은 미래에 받을 것이라 예측되는 보상의 합(value)은 하나의 값으로 예측됩니다. C51 알고리즘은 distributional reinforcement learning 알고리즘입니다. 이 distributional RL 알고리즘은 value를 하나의 scalar 값이 아닌 distribution으로 예측합니다.

<img src="./Images/distributionalRL.png" alt="distributional RL" class = "center" style="width: 500px;"/>

 이에 따라 일반적인 강화학습에서 이용하는 `bellman equation`의 value Q 대신 distribution Z를 사용합니다. 이 bellman equation을 `distributional bellman equation` 이라고 합니다. 해당 식들은 다음과 같습니다. 

 <img src="./Images/bellman_equation.png" alt="distributional RL" class ="center" style="width: 700px;"/>



