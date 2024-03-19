import streamlit as st
from streamlit_option_menu import option_menu
from pages import *


st.set_page_config(layout="wide")

# 옵션에 따라 실행할 함수들을 매핑한 딕셔너리 생성
option_functions = {
    "MAIN" : home,
    "장서현황": lib_status,
    "대출통계": prefer,
    "연령대별 대출통계" : pandas_ai
   
}


# 옵션 메뉴 생성
with st.sidebar:
    selected_option = option_menu("MENU", ["MAIN","장서현황", "대출통계", "연령대별 대출통계"], 
                                icons=['house', 'bi bi-book-fill', "bi bi-book", 'cloud-upload'],
                                default_index=0 
    )


# 선택된 옵션에 맞는 함수 실행
if selected_option in option_functions:
    option_functions[selected_option]()


