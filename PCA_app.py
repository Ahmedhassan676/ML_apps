import streamlit as st
import pandas as pd
import numpy as np
#%% import required packages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats

the_usual_suspects= (TabError,KeyError,AttributeError,UnboundLocalError,TypeError,ValueError,ZeroDivisionError)
def get_df_4m_uploaded_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[1]
    if file_extension == 'xlsx':
        df_uploaded = pd.read_excel(uploaded_file, engine='openpyxl')
    elif file_extension == 'xls':
        df_uploaded = pd.read_excel(uploaded_file)
    elif file_extension == 'csv':
        df_uploaded = pd.read_csv(uploaded_file)
    return df_uploaded
def main_positive():
    html_temp="""
    <div style="background-color:  #11213b  ;padding:16px">
    <h2 style="color:white"; text-align:center> PCA APP </h2>
    </div>
    <style>
    table {
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
    }

    td, th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
    }


    </style>


        """
    st.markdown(html_temp, unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Choose Training Dataset', key = 1)
    
    if uploaded_file:

        #df_uploaded = get_df_4m_uploaded_file(uploaded_file)
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ##                          train PCA model
        ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        #%% fetch data
        #data = get_df_4m_uploaded_file(uploaded_file).iloc[1:,:]

        #%% separate train data
        data_train = get_df_4m_uploaded_file(uploaded_file).iloc[1:,:]
                
        #%% scale data
        scaler = StandardScaler()
        data_train_normal = scaler.fit_transform(data_train)
                
        #%% PCA
        pca = PCA()
        score_train = pca.fit_transform(data_train_normal)

        #%% decide # of PCs to retain and compute reduced data in PC space
        explained_variance = 100*pca.explained_variance_ratio_ # in percentage
        cum_explained_variance = np.cumsum(explained_variance) # cumulative % variance explained

        n_comp = np.argmax(cum_explained_variance >= 90) + 1
        score_train_reduced = score_train[:,0:n_comp]

        st.write('Number of PCs cumulatively explaining atleast 90% variance: ', n_comp)

        #%% reconstruct original data
        V_matrix = pca.components_.T
        P_matrix = V_matrix[:,0:n_comp] 

        data_train_normal_reconstruct = np.dot(score_train_reduced, P_matrix.T)

        #%% calculate T2 for training data
        lambda_k = np.diag(pca.explained_variance_[0:n_comp]) # eigenvalue = explained variance
        lambda_k_inv = np.linalg.inv(lambda_k)

        T2_train = np.zeros((data_train_normal.shape[0],))

        for i in range(data_train_normal.shape[0]):
            T2_train[i] = np.dot(np.dot(score_train_reduced[i,:],lambda_k_inv),score_train_reduced[i,:].T)

        #%% calculate Q for training data
        error_train = data_train_normal - data_train_normal_reconstruct
        Q_train = np.sum(error_train*error_train, axis = 1)

        #%% T2_train control limit
        

        N = data_train_normal.shape[0]
        k = n_comp

        alpha = 0.01# 99% control limit
        T2_CL = k*(N**2-1)*scipy.stats.f.ppf(1-alpha,k,N-k)/(N*(N-k))

        #%% Q_train control limit
        eig_vals = pca.explained_variance_
        m = data_train_normal.shape[1]

        theta1 = np.sum(eig_vals[k:])
        theta2 = np.sum([eig_vals[j]**2 for j in range(k,m)])
        theta3 = np.sum([eig_vals[j]**3 for j in range(k,m)])
        h0 = 1-2*theta1*theta3/(3*theta2**2)

        z_alpha = scipy.stats.norm.ppf(1-alpha)
        Q_CL = theta1*(z_alpha*np.sqrt(2*theta2*h0**2)/theta1+ 1 + theta2*h0*(1-h0)/theta1**2)**2 

        #%% Q_train plot with CL
        fig_train1, ax_train1 = plt.subplots()
        ax_train1.plot(Q_train)
        ax_train1.plot([1,len(Q_train)],[Q_CL,Q_CL], color='red')
        ax_train1.set_xlabel('Sample #')
        ax_train1.set_ylabel('Q for training data')
        st.pyplot(fig_train1)
                
        #%% T2_train plot with CL
        fig_train2, ax_train2 = plt.subplots()
        ax_train2.plot(T2_train)
        ax_train2.plot([1,len(T2_train)],[T2_CL,T2_CL], color='red')
        ax_train2.set_xlabel('Sample #')
        ax_train2.set_ylabel('T$^2$ for training data')
        st.pyplot(fig_train2)

        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ##                          test data
        ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        uploaded_file_test = st.file_uploader('Choose Training Dataset', key = 2)
    
        if uploaded_file_test:
            #%% get test data, normalize it
            data_test = get_df_4m_uploaded_file(uploaded_file_test).iloc[1:,:]
            data_test_normal = scaler.transform(data_test) # using scaling parameters from training data

            #%% compute scores and reconstruct
            score_test = pca.transform(data_test_normal)
            score_test_reduced = score_test[:,0:n_comp]

            data_test_normal_reconstruct = np.dot(score_test_reduced, P_matrix.T)

            #%% calculate T2_test
            T2_test = np.zeros((data_test_normal.shape[0],))

            for i in range(data_test_normal.shape[0]): # eigenvalues from training data are used
                T2_test[i] = np.dot(np.dot(score_test_reduced[i,:],lambda_k_inv),score_test_reduced[i,:].T)

            #%% calculate Q_test
            error_test = data_test_normal_reconstruct - data_test_normal
            Q_test = np.sum(error_test*error_test, axis = 1)

            #%% plot T2_test and T2_train with CL
            T2_combined = np.concatenate([T2_train,T2_test])

            fig1, ax1 = plt.subplots()
            ax1.plot(T2_combined)
            ax1.plot([1,len(T2_combined)],[T2_CL,T2_CL], color='red')
            ax1.plot([69,69],[0,100], color='cyan')
            ax1.set_xlabel('Sample #')
            ax1.set_ylabel('T$^2$ for training and test data')
            st.pyplot(fig1)

            #%% plot Q_test and Q_train with CL
            Q_combined = np.concatenate([Q_train,Q_test])
            fig2, ax2 = plt.subplots()
            ax2.plot(Q_combined)
            ax2.plot([1,len(Q_combined)],[Q_CL,Q_CL], color='red')
            ax2.plot([69,69],[0,100], color='cyan')
            ax2.set_xlabel('Sample #')
            ax2.set_ylabel('Q for training and test data')
            st.pyplot(fig2)


            #track anomalies 

            # T2 Anomaly
            T2_anomaly = [1 if i > T2_CL else 0  for i in T2_combined]
            st.write('Count of T2 anomalies detected {}'.format(sum(T2_anomaly)))
            Q_anomaly = [1 if i > Q_CL else 0  for i in Q_combined]
            st.write('Count of Q anomalies detected {}'.format(sum(Q_anomaly)))
            samples_index_T2 = [i for i in range(len(T2_anomaly)) if T2_anomaly[i] ==1]
            samples_index_Q = [i for i in range(len(Q_anomaly)) if Q_anomaly[i] ==1]
            ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ##                          fault diagnosis by contribution plots
            ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            #%% T2 contribution
            which_anomaly = st.selectbox('Inpect Anomaly',['T2 Anomalies','Q Anomalies'],key='whichanomalypca_34')
            if which_anomaly == 'T2 Anomalies':
                index_list = samples_index_T2
            else: index_list = samples_index_Q
            sample = st.selectbox('Inspect Anomaly: Sample #',index_list,key='aykeyPCAdasd') 
            if sample > N:
                data_point = np.transpose(data_test_normal[sample-N,])
            else:
                data_point = np.transpose(data_train_normal[sample,])

            D = np.dot(np.dot(P_matrix,lambda_k_inv),P_matrix.T)
            T2_contri = np.dot(scipy.linalg.sqrtm(D),data_point)**2 # vector of contributions

            fig3, ax3 = plt.subplots()
            ax3.plot(T2_contri)
            ax3.set_xlabel('Variable #')
            ax3.set_ylabel('T$^2$ contribution plot')
            st.pyplot(fig3)

            T2_top_contributor = np.argmax(T2_contri)
            st.write('Top Contributor is Variable # {}'.format(T2_top_contributor))
            #%% SPE contribution
            if sample > N:
                error_test_sample = error_test[sample-N,]
            else: error_test_sample = error_test[sample,]
            SPE_contri = error_test_sample*error_test_sample # vector of contributions


            fig4, ax4 = plt.subplots()
            ax4.plot(SPE_contri)
            ax4.set_xlabel('Variable #')
            ax4.set_ylabel('SPE contribution plot')
            st.pyplot(fig4)
            Q_top_contributor = np.argmax(SPE_contri)
            st.write('Top Contributor is Variable # {}'.format(Q_top_contributor))

            if which_anomaly == 'T2 Anomalies':
                variable = T2_top_contributor
            else: variable = Q_top_contributor
            #%% variable plot
            fig5, ax5 = plt.subplots()
            ax5.plot(data_test.iloc[:,variable])
            ax5.set_xlabel('Sample #')
            ax5.set_ylabel('Variable # {}'.format(variable))
            st.pyplot(fig5)


if __name__ == '__main__':
    main_positive()