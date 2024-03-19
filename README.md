### 프로젝트 소개

#### 서울도서관 분야별 장서현황과 대출현황 대시보드

- 서울도서관 분야별 장서현황과 대출현황 시각화
    - pandas와 pyecharts, plotly 라이브러리를 활용해 시각화함

- 장서현황
    - 현재 서울도서관 분야별 장서보유 현황을 시각화

- 대출현황
    - 최근3개월자료(24.1.1~24.3.13)
        - 국립중앙도서관으로부터 최근3개월자료를 받아와 도서분야별 누적 대출횟수와 인기대출도서 top10을 보여줌
    - 2021, 2022년
        - selectbox를 통해 2021년과 2022년을 선택하여 분야별 누적 대출현황(%) 시각화
        - 2021년과 2022년의 누적 대출현황 수치의 비교를 위해 두개의 데이터셋의 합친 막대그래프 자료로 1년간의 변화를 시각화

#### 이슈
- 연령대별 대출현황
    - 연령대별 대출현황을 데이터프레임을 기반으로 자동으로 답변을 생성해주는 챗봇형식으로 답변을 주게끔 구상\n
    ex. 총류 대출이 많은 연령대 순위를 알려줘\n
    하지만 openai는 데이터프레임을 인식하지못함 -> pandasai는 데이터프레임 기반 답변 제공가능
    => but, import 오류 생성.. (아직 해결책을 찾지 못함)
    - streamlit에서 pandasai 구현하려 했지만 pandasai 패키지가 설치되었음에도 불구하고 pandasai 내 PandasAI가 import 되지않음
    - streamlit 에서 입력값을 주는 방식이 아니라 코드내에서 질문을 하는 방식으로 운용해보니 질문에 답변을 주는것을 확인하였음