import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import time


col1, col2, col3 = st.beta_columns(3)

with col2:
    st.image("img/logo.png", use_column_width=True)

st.title('TCAS Prediction Model Using KNN')

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data


def get_input():
    # Display widgets and store their values in variables
    v_sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    v_Marital_status = st.sidebar.radio('Married', ['Yes', 'No'])
    v_School = st.sidebar.selectbox('School of', ['School of Agro-industry', 'School of Cosmetic Science', 'School of Dentistry', 'School of Health Science', 'School of Information Technology', 'School of Integrative Medicine',
                                                  'School of Law', 'School of Liberal Arts', 'School of Management', 'School of Medicine', 'School of Nursing', 'School of Science', 'School of Sinology', 'School of Social Innovation'])
    v_Major = st.sidebar.selectbox('Major of', ['Accounting', 'Applied Chemistry', 'Applied Thai Traditional Medicine', 'Aviation Business Management', 'Beauty Technology', 'Biotechnology', 'Business Administration', 'Business Chinese', 'Chinese Language and Culture', 'Chinese Study', 'Computer Engineering', 'Computer Science and Innovation', 'Cosmetic Science', 'Dentistry', 'Digital Technology for Business Innovation', 'Economics', 'English', 'Environmental Health', 'Food Science and Technology', 'Hospitality Industry Management',
                                                'Information Technology', 'International Development', 'Laws', 'Logistics and Supply Chain Management', 'Materials Engineering', 'Medicine', 'Multimedia Technology and Animation', 'Nursing Science', 'Occupational Health and Safety', 'Physical Therapy', 'Posthavest Technology and Logistics', 'Public Health', 'Software Engineering', 'Sports and Health Science', 'Teaching Chinese Language (4 Year)', 'Teaching Chinese Language (5 Year)', 'Tourism Management', 'Traditional Chinese Medicine'])
    v_Tcas = st.sidebar.radio('TCAS type', ['TCAS_TYPE1', 'TCAS_TYPE2', 'TCAS_TYPE3','TCAS_TYPE4','TCAS_TYPE5'])
    v_Region = st.sidebar.selectbox('Region', ['Central', 'East', 'International', 'North', 'NorthEast', 'South', 'West'])
    v_Religion = st.sidebar.selectbox('Religion', ['Bahai', 'Buddhism', 'Christian', 'Hindu','Irreligious', 'Islam', 'Sikhism'])
    v_StudentTH = st.sidebar.radio('Student Thailand', ['Thai','Foreigner'])
    v_GPAX = st.sidebar.slider('GPAX', 0.0, 4.00, 0.01)
    v_GPA_eng = st.sidebar.slider('GPA English', 0.0, 4.00, 0.01)
    v_GPA_math = st.sidebar.slider('GPA Math', 0.0, 4.00, 0.01)
    v_GPA_sci = st.sidebar.slider('GPA Sci', 0.0, 4.00, 0.01)
    v_GPA_sco = st.sidebar.slider('GPA Sco', 0.0, 4.00, 0.01)
    st.sidebar.header('1) Expectation for studying in MFU')
    v_Q1 = st.sidebar.checkbox('beautiful scenary and atmosphere')
    v_Q2 = st.sidebar.checkbox('quality of life')
    v_Q3 = st.sidebar.checkbox('campus and facilities')
    v_Q4 = st.sidebar.checkbox(
        'modern and ready-to-use learning support and facilities')
    v_Q5 = st.sidebar.checkbox('sources of student scholarship')
    v_Q6 = st.sidebar.checkbox('demand by workforce market')
    st.sidebar.subheader('2) Source of information for this application')
    v_Q7 = st.sidebar.checkbox('email')
    v_Q8 = st.sidebar.checkbox('Facebook Admission@MFU')
    v_Q9 = st.sidebar.checkbox('Facebook MFU')
    v_Q10 = st.sidebar.checkbox('Facebook school or major')
    v_Q11 = st.sidebar.checkbox('Road show')
    v_Q12 = st.sidebar.checkbox('Family/friend/relative')
    v_Q13 = st.sidebar.checkbox('school teachers')
    v_Q13 = st.sidebar.checkbox('education exhibitions')
    v_Q14 = st.sidebar.checkbox('tutor schools')
    v_Q15 = st.sidebar.checkbox('television/Youtube channel')
    v_Q16 = st.sidebar.checkbox('advertisement in radio/newspaper/other publications')
    v_Q17 = st.sidebar.checkbox('other sources')
    v_Q18 = st.sidebar.checkbox('https://admission.mfu.ac.th')
    v_Q19 = st.sidebar.checkbox('https://www.mfu.ac.th')
    v_Q20 = st.sidebar.checkbox('other educational websites')
    v_Q21 = st.sidebar.checkbox('telephone/personal contact')
    v_Q22 = st.sidebar.checkbox('easy/convenient transportation')
    st.sidebar.subheader('3) Factor to apply for MFU')
    v_Q23 = st.sidebar.checkbox('suitable cost')
    v_Q24 = st.sidebar.checkbox('graduates with higher language/academic competency than other universities')
    v_Q25 = st.sidebar.checkbox('learning in English')
    v_Q26 = st.sidebar.checkbox('quality/reputation of university')
    v_Q27 = st.sidebar.checkbox('excellence in learning support and facilities')
    v_Q28 = st.sidebar.checkbox('provision of preferred major')
    v_Q29 = st.sidebar.checkbox('environment and setting motivate learning')
    v_Q30 = st.sidebar.checkbox('experienced and high-quality instructors')
    v_Q31 = st.sidebar.checkbox('suggestion by school teacher/friend/relative')
    v_Q32 = st.sidebar.checkbox('suggestion by family')
    st.sidebar.header('4) If your application fails, will you try again?')
    v_Q33 = st.sidebar.checkbox('try the same major')
    v_Q34 = st.sidebar.checkbox('try a different major')
    v_Q35 = st.sidebar.checkbox('will not try again')
    st.sidebar.header('5) Reason for apply for the major')
    v_Q36 = st.sidebar.checkbox('suggestion by school teacher')
    v_Q37 = st.sidebar.checkbox('suggestion by familys')
    v_Q38 = st.sidebar.checkbox('personal preference')
    v_Q39 = st.sidebar.checkbox('chance of getting a job after graduation')
    v_Q40 = st.sidebar.checkbox('less competitive than other universities')
    v_Q41 = st.sidebar.checkbox('suggestion by friend/relative/others')
    v_Q42 = st.sidebar.checkbox('suggestion by friend/relative')

    # Change the value of Student thai to be {'0', '1'} as stored in the trained dataset
    if v_StudentTH == 'Thai':
       v_StudentTH = 1
    elif v_StudentTH == 'Foreigner':
        v_StudentTH = 0
    
    # Change the value of Marital Status to be {'0', '1'} as stored in the trained dataset
    if v_Marital_status == 'Yes':
        v_Marital_status = 1
    elif v_Marital_status == 'No':
        v_Marital_status = 0

    # Change the value of TCAS type to be {'type1', 'type2', 'type3', 'type4', 'type5'} as stored in the trained dataset
    if v_Tcas == '1':
        v_Tcas = 'TCAS_TYPE1'
    elif v_Tcas == '2':
        v_Tcas = 'TCAS_TYPE2'
    elif v_Tcas == '3':
        v_Tcas = 'TCAS_TYPE3'
    elif v_Tcas == '4':
        v_Tcas = 'TCAS_TYPE4'
    elif v_Tcas == '5':
        v_Tcas = 'TCAS_TYPE5'

    # Store user input data in a dictionary
    data = {
            'Sex': v_sex,
            'MaritalStatus': v_Marital_status,
            'FacultyName': v_School,
            'DepartmentName': v_Major,
            'TCAS': v_Tcas,
            'HomeRegion': v_Region,
            'ReligionName': v_Religion,
            'StudentTH' : v_StudentTH,
            'GPAX': v_GPAX,
            'GPA_Eng': v_GPA_eng,
            'GPA_Math': v_GPA_math,
            'GPA_Sci': v_GPA_sci,
            'GPA_Sco': v_GPA_sco,
            'Q1': v_Q1,
            'Q2': v_Q2,
            'Q3': v_Q3,
            'Q4': v_Q4,
            'Q5': v_Q5,
            'Q6': v_Q6,
            'Q7': v_Q7,
            'Q8': v_Q8,
            'Q9': v_Q9,
            'Q10': v_Q10,
            'Q11': v_Q11,
            'Q12': v_Q12,
            'Q13': v_Q13,
            'Q14': v_Q14,
            'Q15': v_Q15,
            'Q16': v_Q16,
            'Q17': v_Q17,
            'Q18': v_Q18,
            'Q19': v_Q19,
            'Q20': v_Q20,
            'Q21': v_Q21,
            'Q22': v_Q22,
            'Q23': v_Q23,
            'Q24': v_Q24,
            'Q25': v_Q25,
            'Q26': v_Q26,
            'Q27': v_Q27,
            'Q28': v_Q28,
            'Q29': v_Q29,
            'Q30': v_Q30,
            'Q31': v_Q31,
            'Q32': v_Q32,
            'Q33': v_Q33,
            'Q34': v_Q34,
            'Q35': v_Q35,
            'Q36': v_Q36,
            'Q37': v_Q37,
            'Q38': v_Q38,
            'Q39': v_Q39,
            'Q40': v_Q40,
            'Q41': v_Q41,
            'Q42': v_Q42}

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df

def ConvertSex(x):
    if x == "Female":
        return 1
    elif x == "Male":
        return 0

# -- Call function to display widgets and get data from user
df = get_input()
st.header('Web Application of MFU Prediction')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('tcas_simpledata.csv')
df = pd.concat([df, data_sample], axis=0)

# Delete MaritalStatus form numerical_data
numerical_data = df.select_dtypes(include=['number'])
numerical_data = numerical_data.drop(columns=['MaritalStatus'])

# One-hot encoding for nominal features
cat_data_school = pd.get_dummies(df['FacultyName'])
cat_data_mojor = pd.get_dummies(df['DepartmentName'])
cat_data_tcas = pd.get_dummies(df['TCAS'])
cat_data_home = pd.get_dummies(df['HomeRegion'])
cat_data_reli = pd.get_dummies(df['ReligionName'])

# Select Sex and MaritalStatus form df
df['Sex'] = df.Sex.apply(ConvertSex)
Sex_data = df[['Sex']]
Marital_data = df[['MaritalStatus']]

# Combine all transformed features together
X_new = pd.concat([Sex_data, Marital_data, cat_data_school, cat_data_mojor,cat_data_tcas, cat_data_home,cat_data_reli, numerical_data], axis=1)
X_new = X_new[:1]  # Select only the first row (the user input data)

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X_new)

# -- Reads the saved normalization model
load_sc = pickle.load(open('normalization.pkl', 'rb'))
# Apply the normalization model to new data
X_new = load_sc.transform(X_new)

# -- Display normalized new data:
st.subheader('Normalized Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)

# -- Display predicted class:
st.subheader('Prediction:')

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1)
st.success('Done!')
st.balloons()

#prediction = [ mfu student , Not mfu student ])
st.write(prediction)
st.stop()