from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from django.http import HttpResponse
from .models import *
import pandas as pd
import pickle
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create your views here.

def index(request):
    return render(request, 'health/index.html')

def prediction(request):
    age = int(request.POST['age'])
    sex = int(request.POST['gender'])
    ht = int(request.POST['ht'])
    wt = int(request.POST['wt'])
    waist = float(request.POST['waist'])
    smoking = float(request.POST['smoking'])
    alcohol = float(request.POST['alcohol'])
      
    age1 = (age//5) + 1
    if (ht%10)<5:
        ht = ht - (ht%10)
    else:
        ht = ht - (ht%10) + 5
    if (wt%10)<5:
        wt = wt - (wt%10)
    else:
        wt = wt - (wt%10) + 5
    bmi = float(wt) / ((float(ht)/100)**2)
    absi = (waist*0.393701) /((bmi**0.6666) * ((float(ht)/100)**0.5))
    bmi = round(bmi, 2)
    absi = round(absi, 2)

    new_info = Information(sex = sex, age = age, ht = ht, wt = wt, waist = waist, smoking = smoking, alcohol = alcohol, bmi = bmi, absi = absi)
    new_info.save()


    X_test = pd.DataFrame([[sex, age1, ht, wt, bmi, absi, waist, smoking, alcohol]],columns = ['sex','age','ht','wt','bmi','absi','waist','smoking','alcohol'])
    test = PolynomialFeatures(degree=3).fit_transform(X_test)
    lgbm_sbp = pickle.load(open('C:/Users/sja95/HealthWeb/health/LGBM_SBP.sav', 'rb'))
    lgbm_dbp = pickle.load(open('C:/Users/sja95/HealthWeb/health/LGBM_DBP.sav', 'rb'))
    pl3_fbs = pickle.load(open('C:/Users/sja95/HealthWeb/health/PL3_FBS.sav', 'rb'))
    y_pred_sbp = lgbm_sbp.predict(X_test)
    y_pred_dbp = lgbm_dbp.predict(X_test)
    y_pred_fbs = pl3_fbs.predict(test)

    y_pred_sbp = int(y_pred_sbp[0])
    y_pred_dbp = int(y_pred_dbp[0])
    y_pred_fbs = int(y_pred_fbs[0])


    if age<=24:
        age2 = 1
    else:
        age2 = ((age-1)//2)-10
        
    ppr = y_pred_sbp - y_pred_dbp
    X_test2 = pd.DataFrame([[sex, age2, y_pred_fbs, bmi, ppr]], columns = ['sex','age','fbs','bmi', 'ppr'])
    lgbm_bp = pickle.load(open('C:/Users/sja95/HealthWeb/health/LGBM_bp.sav', 'rb'))
    y_pred_bpd = (lgbm_bp.predict(X_test2))[0]

    fbs2 = y_pred_fbs ** 2
    X_test3 = pd.DataFrame([[age2,sex,y_pred_sbp, y_pred_dbp, bmi, y_pred_fbs, fbs2]], columns = ['age','sex','sbp','dbp','bmi','fbs','squared_fbs'])
    lgbm_fbs = pickle.load(open('C:/Users/sja95/HealthWeb/health/LGM_FBSD.sav', 'rb'))
    y_pred_fbsd = (lgbm_fbs.predict(X_test3))[0]

    # 결과 종합
    # 고혈압 
    # sbp가 120이상 140이하 | 진단 1 => 주의 
    # sbp가 141이상 | 진단 1 =>  위험

    if y_pred_bpd==1:
        if (y_pred_sbp > 140)|(y_pred_dbp > 90) : 
            result_bp = '위험'
        elif (y_pred_sbp >= 120 )|(y_pred_dbp < 90) :
            result_bp = '주의'
        else:
            result_bp = '정상'
    else:
        if (y_pred_sbp > 140)&(y_pred_dbp >= 90) : 
            result_bp = '위험'
        elif (y_pred_sbp >= 120 )&(y_pred_dbp < 90) :
            result_bp = '주의'
        else:
            result_bp = '정상'

    # 당뇨
    # fbs가 100이상  125이하 | 진단 Y => 주의
    # fbs  126이상 | 진단 Y => 위험

    if y_pred_fbsd =='Y':
        if y_pred_fbs >= 128 :
            result_fbs = '위험'
        else :
            result_fbs = '주의'
    else :
        if y_pred_fbs >= 128:
            result_fbs = '위험'
        elif y_pred_fbs >= 100 :
            result_fbs = '주의'
        else:
            result_fbs = '정상'

    if (result_fbs == '정상') & (result_bp == '정상') :
        result_message = '건강 관리를 잘 하시고 있군요!'
    elif (result_fbs == '위험') | (result_bp == '위험') :
        result_message = '전문의와의 상담이 필요해 보여요!'
    else:
        result_message = '흠... 건강검진 한 번 받아보시면 어때요?'

    if age >= 60 :
        if (result_bp == '주의')|(result_bp == '위험') :
            recommend_message = '마그네슘, 비타민C로 혈압 건강 제대로 잡자!'
            case = 2
        elif (result_fbs == '주의')|(result_fbs == '위험') :
            recommend_message = '혈당 관리는 마그네슘과 코로솔산으로 시작!'
            case = 1
        else :
            recommend_message = '몸도 마음도 모두 이팔청춘! 오메가3로 더욱 더 젊게!'
            case = 3
    elif age >= 40 :
        if (result_fbs == '주의')|(result_fbs == '위험') :
            recommend_message = '코로솔산, 미네랄로 혈당 관리를 더 쉽게!'
            case = 4
        elif (result_bp == '주의')|(result_bp == '위험') :
            recommend_message = '마그네슘, 종합비타민으로 혈압은 내리고! 건강은 올리고!'
            case = 5
        else :
            recommend_message = '미네랄, 항산화제와 함께 지금 건강 그대로 쭉-!'
            case = 6
    else :
        if (result_bp == '주의')|(result_bp == '위험') :
            recommend_message = '미네랄과 종합비타민으로 혈압 관리 끝! '
            case = 8
        elif (result_fbs == '주의')|(result_fbs == '위험') :
            recommend_message = '마그네슘, 종합비타민으로 혈당 초기에 잡자!'
            case = 7
        else :
            recommend_message = '학업과 업무로 피곤한 나를 위해 비타민B FLEX!'
            case = 9


    context = {
        'sbp_pred':y_pred_sbp, 
        'dbp_pred':y_pred_dbp, 
        'fbs_pred':y_pred_fbs, 
        'result_fbs':result_fbs,
        'result_bp':result_bp,
        'result_message' : result_message,
        'recommend_message' : recommend_message,
        'case' :case
    }

    # 모델 예측하고 값을 result.html로 보내주면 됨. result화면이 결과화면!
    return render(request, 'health/result.html', context)





   
    
