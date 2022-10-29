import streamlit as st

    
# def predicitions(X_test,y_test):



#     fp = eur.history(interval='1m',period='60m')
#     fp.Close.values

#     X=[]
#     # end=3
#     x=fp.Close.values

#     for i in range(len(x)):
#         end+=1
#         if end>=len(x)-1:break
#         X.append(x[i:end])
#     #     Y.append(x[end])

#     # import numpy as np
#     new_points=np.array(X).reshape(np.array(X).shape[0], 4,1)
#     new_points.shape

#     pred1=nn.predict(new_points)

#     fig1,ax1=plt.subplots()
#     ax1.plot(x,label='real')
#     ax1.plot(pred1)
#     ax1.legend()
#     st.pyplot(fig1)