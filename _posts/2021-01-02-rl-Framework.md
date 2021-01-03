---
title: RL opensource framework
layout: post
date: "2021-01-03 12:00:00"
author: deepwonwoo
tags: rl opensource
cover: "/assets/keyboard2.jpg"
categories: rl
---





딥러닝에 tensorflow, pytorch등이 있듯이 강화학습에도 여러 Framework들이 있다.

물론 배운 이론을 Scratch부터 구현하는것이 많은 공부가 될테지만, 그 시간에 그대보다 뛰어난 분들이 만든 코드를 분석하고 더 깊게 공부하는것이 좋다고 생한다. (Opensource 시대 아닌가)

아래 표는 Catalyst team의 [Sergey Kolesnikov](https://twitter.com/Scitator)가 [spreadsheet]( https://docs.google.com/spreadsheets/d/1ZWhViAwCpRqupA5E_xFHSaBaaBZ1wAjO6PvmmEEpXGI/edit#gid=812627647)에 정리한 RL opensource들을 참고하여 추가 작성한 것이다. 

|                           Project                            | Maintainer                  | Framework  | Stars   | Rainbow | PPO  | SAC  |
| :----------------------------------------------------------: | --------------------------- | ---------- | ------- | ------- | ---- | ---- |
|    [Catalyst](https://github.com/catalyst-team/catalyst)     | Sergey Kolesnikov           | PyTorch    | 2349    | ❌       | ✔️    | ✔️    |
|   [OpenAI baselines](https://github.com/openai/baselines)    | OpenAI                      | Tensorflow | 10958   | ❌       | ✔️    | ❌    |
|   [baselines](https://github.com/hill-a/stable-baselines)    | Antonin Raffin, Ashley Hill | Tensorflow | 2734    | ❌       | ✔️    | ✔️    |
| [Ray.rllib](https://github.com/ray-project/ray/tree/master/python/ray/rllib) | Ray Team                    | Tensorflow | 14250   | ✔️       | ✔️    | ✔️    |
|      [TF agents](https://github.com/tensorflow/agents)       | Google                      | Tensorflow | 1740    | ❌       | ✔️    | ✔️    |
|    [Horizon](https://github.com/facebookresearch/Horizon)    | Facebook                    | PyTorch    | 2733    | ❌       | ✔️    | ✔️    |
|       [Coach](https://github.com/NervanaSystems/coach)       | Intel                       | Tensorflow | 1886    | ✔️       | ✔️    | ✔️    |
|        [Dopamine](https://github.com/google/dopamine)        | Google                      | Tensorflow | 9252    | ✔️       | ❌    | ❌    |
|        [ELF](https://github.com/facebookresearch/ELF)        | Facebook                    | PyTorch    | 2011    | ❌       | ❌    | ❌    |
|       [keras-rl](https://github.com/keras-rl/keras-rl)       | Matthias Plappert           | Tensorflow | 4904    | ❌       | ❌    | ❌    |
|        [Rainbow](https://github.com/Kaixhin/Rainbow)         | Kai Arulkumaran             | PyTorch    | 1078    | ✔️       | ❌    | ❌    |
|       [Garage](https://github.com/rlworkgroup/garage)        | rlworkgroup                 | Tensorflow | 1008    | ❌       | ✔️    | ✔️    |
|                [Reagent](https://reagent.ai/)                | facebook                    | PyTorch    | 2700    | ❌       | ✔️    | ✔️    |
|           [Acme](https://github.com/deepmind/acme)           | deepmind                    | Tensorflow | 1800    | ❌       | ❌    | ❌    |
|  [Tensorforce](https://github.com/tensorforce/tensorforce)   | Google                      | Tensorflow | 2800    | ❌       | ✔️    | ❌    |
|        [RLgraph](https://github.com/rlgraph/rlgraph)         | RLgraph                     | pytorch    | 273     | ❌       | ✔️    | ✔️    |
|            [PFRL](https://github.com/pfnet/pfrl)             | pfnet                       | pytorch    | 469     | ✔️       | ✔️    | ✔️    |
| [SOTA-RL-Algorithms](https://github.com/deepwonwoo/SOTA-RL-Algorithms) | quantumiracle               | pytorch    | 218     | ❌       | ✔️    | ✔️    |
| [rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need) / [pg-is-all-you-need](https://github.com/MrSyee/pg-is-all-you-need) | Curt, kyunghwan             | pytorch    | 869/324 | ✔️       | ✔️    | ✔️    |



요즘 딥러닝을 할때면, data가 image라면 CNN계열, timeseries라면 rnn계열을 쓰는게 일반적이고(요즘은 attention이 다 씹어먹고 있는듯 하지만..),  Framework도 특별한 이유가 아니고서야 Pytorch아니면 tensorflow를 쓴다.

강화학습은 머신러닝에서도 가장 발전되지 않았고 아직 학계에서도 활발히 연구중인 분야이기 때문에 아직 잘 알려진 rule of thumb이 없다.  (논문들을 보면 검증은 커녕 benchmark도 다 제각각이고 용어, 알고리즘들도 체계적이지 않다. 그렇기에 참 매력적이고 가능성이 많은 분야인거 같다.)

RL Framework들도 찾아보면 초기 딥러닝때와 같이 굉장히 다양하며, 각기 다른 철학과 용도에 따라 각 기관에서 개발되고 배포되고 있다. 따라서 자신에게 맞는 취향과 목적에 맞는 Framework를 선택하여 사용하기를 추천한다.

아래는 필자가 직접 사용해보았던 Framework들의 느낀점을 적어 보았다.



## [Ray.rllib](https://github.com/ray-project/ray/tree/master/python/ray/rllib)

* [RLlib](https://docs.ray.io/en/master/rllib.html)은 분산 컴퓨팅을위한 Python 라이브러리 인 [Ray의](https://ray.io/) 일부이다. Ray는 2018년 UC Berkeley의 [RISELab](https://rise.cs.berkeley.edu/)에서 개발된 프로젝트이며, 이 연구실에서 Apache Spark가 만들어 졌다. Ray Core를 기반으로 RLlib뿐만 아니라 하이퍼파라미터 최적화 라이브러리라 [Tune](https://docs.ray.io/en/master/tune/index.html), Scalable Ray를 위한 [Ray Cluster](https://docs.ray.io/en/master/cluster/index.html), 분산 딥러닝 라이브러리인 [RaySGD](https://docs.ray.io/en/master/raysgd/raysgd.html) 등 여러 기능들이 있다.  이 연구실 출신 핵심 개발자들이 [Anyscale](https://www.anyscale.com/)이라는 회사를 차려 RAY에 대한 교육 및 기능 업그레이드를 통해 영향력을 계속 확장하고 있다.

* 현재 가장 앞서가는 rl 프레임워크라 보인다. 주변에서 ray.rllib을 많이 추천하고 실제로 현업에서도 사용하고있다.
* 단점으로는 Ray Core를 이해해야하며 개인용도에 맞게 Agent를 cutomizing하기 어렵다고 한다.



## [Catalyst](https://github.com/catalyst-team/catalyst)

* 2018년부터  [Sergey Kolesnikov](https://twitter.com/Scitator) 가 만들고 있는 딥러닝 연구개발용 pipeline 프레임워크이다. 여러 용도에 맞는 표준 pipeline을 제공하고 있고, 강화학습용 pipeline [Catalyst.RL](https://github.com/Scitator/catalyst-rl-framework)도 꾸준히 업데이트 중이다. 꾸준히 NeurIPS tutorial/workshop 섹션에 참가하고 있다.
* Sergey Kolesnikov의 [유튜브](https://youtu.be/1g6BpItJdJA) 인터뷰를 보면 Catalyst.rl에 대한 자부심을 느낄수있다. (역시 강화학습의 고향은 러시아라는 건가...)
* 단점으로는 역시 catalyst project 전반의 철학을 이해해야 하고 쓰는 사람이 많지 않다...



## [keras-rl](https://github.com/keras-rl/keras-rl)

* 필자가 썼던 첫번째 강화학습 Framework이다. keras밖에 모를때 keras로 구현한 github repository를 찾다가 발견했었는데, 코드 구성이 직관적이여서 customizing 하기도 좋다.
* 단점이라면 구현된 알고리즘들이 몇개 없는데 업데이트도 잘 안해주는거 같다.



## [SOTA-RL-Algorithms](https://github.com/deepwonwoo/SOTA-RL-Algorithms)

* 다양한 버전의 SAC, PPO 가 구현되어 있으며, 병렬화 코드도 있다. 
* 프레임워크라기 보단 그냥 구현체들을 모아둔거여서 직관적으로 구현되어 있다. 핵심모듈들만 잘 분석해서 내  용도에 맞게 커스터마이징해서 쓰기 편하다. 강추한다.

