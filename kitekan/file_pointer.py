
def get_filename(username,data_type):
    #data_type = "photo"
    MAIN_PATH="/media/kazumi/4b35d6ed-76bb-41f1-9d5d-197a4ff1a6ab/backup/home/kazumi/mogura/"
    user_main_path=MAIN_PATH+"/"+username+"/"

    if(data_type=="photo"):
      datafile=user_main_path+"ad_all.csv"
    elif(data_type=="imu"):
      datafile=user_main_path+"/imu_all.csv"

    #filedata,rate_label=file_pointer.get_filename(data_type)

    #filedata = "../1110/new_exp/21091010-175046/imu_data_21091010-175110.csv"#imu
    #datafile= "../1110/ad_all.csv"#photo and ad

    rate_label=data_type+"_rate"
    return datafile,rate_label
