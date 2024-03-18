# import os
# from dotenv import load_dotenv
# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# from pandasai import PandasAI
import streamlit as st
import pandas as pd
from pyecharts.charts import Bar,Pie
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts
import plotly.graph_objects as go


def home():
    st.markdown("## :books: 서울도서관 분야별 장서현황과 대출현황 대시보드 :books:")
    st.text("서울시 열린데이터 광장에서 공공데이터 정보 받음")
    st.page_link('https://data.seoul.go.kr/', label='서울시 열린데이터 광장 홈페이지')

    st.divider()

    st.markdown("#### 대시보드 기획내용")
    st.markdown("###### 도서관 분야별 도서 보유 현황과 대출현황을 차트를 통해 시각화한 후 이 자료를 바탕으로 챗봇에게 추가 구입하는 도서 분야 추천받는 대시보드를 기획하였습니다")

# 장서현황 page
def lib_status():
    df = pd.read_excel('서울도서관 분야별 장서현황_2022.12.31.기준.xlsx')

    # 데이터 추출
    df = df.iloc[:,2:-2]
    new_df = pd.concat([df.iloc[0], df.iloc[2]], axis=1).T
    new_columns = new_df.iloc[0].astype(str).str.replace('\n','').tolist()
    new_df.columns = new_columns
    new_df.drop(0)
    new_df = new_df.reset_index(drop=True)

    ## 막대그래프로 그리기
    # 리스트로 변경
    x = new_columns
    y = new_df.iloc[1].tolist()

    bar_chart = (
        Bar()
        .add_xaxis(x)
        .add_yaxis("분류",y, label_opts=False)
        .set_global_opts(title_opts=opts.TitleOpts(title = "서울도서관 분야별 장서보유 현황( 수치 )",
                                                    subtitle= "마우스를 막대 위에 올리면 정확한 수치 확인이 가능합니다"))       
    )

    ## 파이차트로 그리기
    # 파이 그래프에 사용될 데이터셋
    z = [list(i) for i in zip(x,y)]

    pie_chart = (
        Pie()
        .add("",z, radius=["30%","80%"]) # radius : 내부 및 외부 링 크기
        .set_global_opts(title_opts=opts.TitleOpts(title="서울도서관 분야별 장서보유 현황( % 비율 )",subtitle="마우스를 원 위에 올리면 정확한 수치 확인이 가능합니다"),
                         legend_opts=opts.LegendOpts(orient="vertical", pos_top="20%", pos_left="5%")) # 범례 조정
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : ({d}%)"))
    )

    st_pyecharts(bar_chart)
    st_pyecharts(pie_chart)


# 대출누적 page
def prefer():
    
    loan_file_paths = ['서울도서관 도서분야별성별 대출 통계_2021.xlsx','서울도서관 도서분야별성별 대출 통계_2022.xlsx']
    loan_total = []

    # 2021 = loan_total[0], 2022 = loan_total[1]
    for file_path in loan_file_paths:
        df = pd.read_excel(file_path,header=1)
        loan_total.append(df)

    # 2021년 데이터 추출
    df_2021 = loan_total[0].iloc[:,3:-2]
    last_row = df_2021.iloc[-1]
    sorted_values = last_row.sort_values(ascending=False)
    sorted_rank_2021 = sorted_values.to_frame().transpose()

    # 2022년 데이터 추출
    df_2022 = loan_total[1].iloc[:,3:-2]
    last_row = df_2022.iloc[-1]
    sorted_values = last_row.sort_values(ascending=False)
    sorted_rank_2022 = sorted_values.to_frame().transpose()

    select_tabs = ["최근 3개월 자료", "2021년, 2022년 자료"]
    select_boxs = ["2021년", "2022년", "2021년 2022년 전체"]
    
    tab1, tab2 = st.tabs(select_tabs)

    with tab1:
        prefer_loan = pd.read_csv('서울도서관 인기대출 도서목록 100선 정보.csv',encoding='cp949')
        prefer_loan['분류기호'].value_counts()
        category_mapping = {
            0:'총류',
            1:'철학',
            2:'종교',
            3:'사회과학',
            4:'자연과학',
            5:'기술과학',
            6:'예술',
            7:'언어',
            8:'문학',
            9:'역사'
        }
        prefer_loan['분류기호'] = prefer_loan['분류기호'].map(category_mapping)
        category_value = prefer_loan['분류기호'].value_counts().sort_index(ascending=False)
                
        # 그래프 그리기
        x = category_value.index.tolist()
        y = category_value.values.tolist()

        category_chart = (
            Bar()
            .add_xaxis(x)
            .add_yaxis("장르별",y)
            .set_global_opts(title_opts = opts.TitleOpts(title = "도서분야별 누적 대출횟수"))
        )

        st_pyecharts(category_chart)

        st.divider()

        st.markdown("##### 인기대출 도서 top10")

        # Figure 생성
        fig = go.Figure()

        top_10_books = prefer_loan.iloc[:10]

        # 테이블 생성
        fig.add_trace(go.Table(
            header = dict(values = list(top_10_books.columns[1:4]),
                          fill_color = 'paleturquoise',
                          align='left'),
            cells=dict(values=[top_10_books.제목, top_10_books.저자, top_10_books.발행처],
                       fill_color = 'white',
                       align='left')
        ))

        st.write(fig)

        

    with tab2:
        choose_select = st.selectbox("년도 선택(2021,2022)", select_boxs, index=None)

        if choose_select == select_boxs[0]:

            # 파이차트
            x_2021 = sorted_rank_2021.columns.tolist()
            y_2021 = sorted_rank_2021.values.flatten().tolist()
            z = [list(i) for i in zip(x_2021,y_2021)]

            pie_chart_2021 = (
                Pie()
                .add("",z, radius=["30%","80%"], center=['50%', '55%']) # radius : 내부 및 외부 링 크기
                .set_global_opts(title_opts=opts.TitleOpts(title="2021년 분야별 누적 대출 ( % )",subtitle="마우스를 원 위에 올리면 정확한 수치 확인이 가능합니다"),
                         legend_opts=opts.LegendOpts(orient="vertical", pos_top="20%", pos_left="5%")) # 범례 조정
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : ({d}%)"))
            )

            st_pyecharts(pie_chart_2021)

        elif choose_select == select_boxs[1]:

            # 파이차트
            x_2022 = sorted_rank_2022.columns.tolist()
            y_2022 = sorted_rank_2022.values.flatten().tolist()
            z = [list(i) for i in zip(x_2022, y_2022)]

            pie_chart_2021 = (
                Pie()
                .add("",z, radius=["30%","80%"], center=['50%', '55%']) # radius : 내부 및 외부 링 크기
                .set_global_opts(title_opts=opts.TitleOpts(title="2022년 분야별 누적 대출 ( % )",subtitle="마우스를 원 위에 올리면 정확한 수치 확인이 가능합니다"),
                         legend_opts=opts.LegendOpts(orient="vertical", pos_top="20%", pos_left="5%")) # 범례 조정
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b} : ({d}%)"))
            )

            st_pyecharts(pie_chart_2021)

        elif choose_select == select_boxs[2]:

            x_total = sorted_rank_2022.columns.tolist()
            y_2021_total = sorted_rank_2021.values.flatten().tolist()
            y_2022_total = sorted_rank_2022.values.flatten().tolist()

            double_bar = (
                Bar()
                .add_xaxis(x_total)
                .add_yaxis("2021년",y_2021_total, gap="0%", label_opts=False)
                .add_yaxis("2022년",y_2022_total, gap="0%", label_opts=False)
                .set_global_opts(title_opts=opts.TitleOpts(title= "2021년과 2022년 분야별 누적 대출 수치 비교",
                                                           subtitle="각각의 막대위에 마우스를 올리면 정확한 수치 확인이 가능합니다"))
            )

            st_pyecharts(double_bar)

def pandas_ai():
    """
    기존기획은 아래의 1번 코드로 pandas ai 라이브러리를 활용해 데이터프레임 값을 기반으로
    사용자의 질문에 대답해주는 챗봇을 기획하였으나 import 오류가 생성되어
    (pandasai 패키지가 설치되었음에도 불구하고 PandasAI가 import 되지않음)
    2번의 코드로 streamlit에 구현되진 않지만
    함수 내에서 질문하는 것에 답변을 주는 것을 확인함

    """

    ### 1. steamlit chatbot 기획 = > import 오류

    # load_dotenv()
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # llm = OpenAI(api_token=openai_api_key)
    # pandas_ai = PandasAI(llm)

    # st.title("Pandas AI")

    # uploaded_file = st.file_uploader("upload your CSV file", type=['csv'])

    # if uploaded_file is not None:
    #     df = pd.read_csv(uploaded_file)
    #     st.write(df.head())

    #     prompt = st.text_area("Enter your prompt: ")

    #     if st.button("generate"):
    #         if prompt:
    #             st.write("pandasai is generate an answer, please wait")
    #             st.write(pandas_ai.run(df, prompt=prompt))
    #         else:
    #             st.warning("please enter a prompt")


    """"""

    # 2. streamlit 구현실패 후, SmartDataframe 이용해 답변주는 형식 구상
    # -> 모든 질문에 답변을 주는 것은 아니지만, 약간의 답변을 줌

    # load_dotenv()
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # llm = OpenAI(api_token=openai_api_key)

    # df = pd.read_excel('서울도서관 도서분야별성별 대출 통계_2021.xlsx')
    # new_columns = df.iloc[0].values
    # df.columns = new_columns.astype(str)

    # df.drop(index = 0, columns= df.columns[0], inplace=True)

    # sums = []
    # for i in range(7):
    #     row1 = df.iloc[i,1:]
    #     row2 = df.iloc[i+8, 1:]
    #     row_sum = row1 + row2
    #     sums.append(row_sum)

    # result_df = pd.DataFrame(sums)

    # result_df.insert(0,'연령대', df.iloc[:7,0].values)
    # result_df.insert(0,'년도','2021')


    # agent = SmartDataframe(result_df,config={"llm":llm})

    # print(agent.chat('총류 연령대 순위를 알려줘'))
    # print(agent.chat("20대 연령대 안에서 높은 숫자의 이름대로 나열해줘"))


    txt = """pandas ai 라이브러리 이용하여 연령대별 누적 대출 분야가 많은 순위를 
    사용자가 질문을 하면 pandasai가 대답해주는 챗봇형식으로 하려하였으나 streamlit에 구현실패하였습니다.
    pandas ai를 이용한 질의응답 코드는 pages.py 파일 내 주석으로 달아놓았습니다"""
    
    st.text(txt)

