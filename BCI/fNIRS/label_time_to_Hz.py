import pandas as pd


def deal(input_list):
    # list转dataframe
    df = pd.DataFrame([], columns=input_list)
    print(df)
    # 保存到本地excel
    #df.to_excel("company_name_li.xlsx", index=False)
    return df



for i in range(1,30):
    for a in range(2,7,2):
        # 读取 Excel 文件
        input_file = '/root/autodl-tmp/project/Open_Access_BCI_Data/fNIRS_data/label_time/MA/sub0'+str(i)+'_session0'+str(a)+'_label_time.xlsx'  # 输入文件名
        output_file = '/root/autodl-tmp/project/Open_Access_BCI_Data/fNIRS_data/label_Hz/MA/sub0'+str(i)+'_session0'+str(a)+'_label_Hz.xlsx'  # 输出文件名

        # 使用 pandas 读取 Excel 文件
        df = pd.read_excel(input_file)

        # 对所有数据乘以 0.01，并转化为整数
        arr = list(df.columns)
        arr_1 = [i * 0.01 for i in arr]
        arr_2 = list(map(int,arr_1))

        # 将结果保存到新的 Excel 文件
        deal(arr_2).to_excel(output_file, index=False)

        print(f"数据已成功处理并保存到 {output_file}")


