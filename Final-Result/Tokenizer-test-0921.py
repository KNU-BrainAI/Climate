# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:25:56 2021

@author: puran


#
text[1] 
유전정보를 활용한 새로운 해충 분류군 동정기술 개발 가  외래 및 돌발해충의 발생조사 및 종 동정 ○ 대상해충   최근 새롭게 발견된 돌발 및 외래해충  나  외래 및 돌발해충의 분포확산 모니터링 ○ 대상해충    가 의 돌발 및 외래해충  다  외래 및 돌발해충의 유전적 다양성 조사 ○ 시험곤충    나 의 해충별 전국 단위 채집표본○ 새로운 해충분류군의 동정기술 개발 및 유입확산 추적뉴클레오티드 염기서열  분자마커  종 동정  침샘  전사체
"""


import torch 

from torch.nn import functional as F
from tokenization_kobert import KoBertTokenizer
from kobert_transformers import get_tokenizer



from konlpy.tag import Mecab
mecab = Mecab('C:/mecab/mecab-ko-dic')

#tokenizer_kb = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일
tokenizer = get_tokenizer()

text_1 ="유전정보를 활용한 새로운 해충 분류군 동정기술 개발 가  외래 및 돌발해충의 발생조사 및 종 동정 ○ 대상해충   최근 새롭게 발견된 돌발 및 외래해충  나  외래 및 돌발해충의 분포확산 모니터링 ○ 대상해충    가 의 돌발 및 외래해충  다  외래 및 돌발해충의 유전적 다양성 조사 ○ 시험곤충    나 의 해충별 전국 단위 채집표본○ 새로운 해충분류군의 동정기술 개발 및 유입확산 추적뉴클레오티드 염기서열  분자마커  종 동정  침샘  전사체"

#temp_X = mecab.nouns(text[i]) # 토큰화

kb_tk = tokenizer.tokenize(text_1)
mecab_nouns = mecab.nouns(text_1)

#list의 element들을 공백을 이용해서 결합
mecab_nouns_join = ' '.join(mecab_nouns) 

mecab_morphs = mecab.morphs(text_1)

#list의 element들을 공백을 이용해서 결합
mecab_morphs_join = ' '.join(mecab_morphs) 


mecab_and_kb_tokenize = tokenizer.tokenize(mecab_nouns_join)


mecab_and_kb_join = ' '.join(mecab_and_kb_tokenize)
