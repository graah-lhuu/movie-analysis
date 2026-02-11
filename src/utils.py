# src/utils.py
"""
工具函数模块
包含数据加载、清洗和可视化辅助函数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
def setup_visualization():
    """设置可视化参数"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    print("✅ 可视化环境设置完成")

def load_data(data_path=None):
    """加载电影数据"""
    if data_path is None:
        # 默认路径
        data_path = "../data/raw/movie_metadata.csv"
    
    print(f"📂 正在从 {data_path} 加载数据...")
    try:
        df = pd.read_csv(data_path)
        print(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
        return df
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def clean_movie_data(df):
    """清洗电影数据"""
    df_clean = df.copy()
    
    print("🧹 开始数据清洗...")
    
    # 1. 查看基本信息
    print(f"原始数据形状: {df_clean.shape}")
    print(f"列名: {list(df_clean.columns)}")
    
    # 2. 处理电影标题（去除首尾空格和特殊字符）
    if 'movie_title' in df_clean.columns:
        df_clean['movie_title'] = df_clean['movie_title'].astype(str).str.strip()
        print("✅ 电影标题已清理")
    
    # 3. 删除完全重复的行
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    print(f"✅ 删除了 {removed_duplicates} 个重复记录")
    
    # 4. 处理缺失值
    print("\n🔍 缺失值统计:")
    missing_stats = df_clean.isnull().sum()
    missing_percent = (missing_stats / len(df_clean) * 100).round(2)
    missing_df = pd.DataFrame({
        '缺失数量': missing_stats,
        '缺失百分比%': missing_percent
    })
    display(missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False))
    
    # 5. 处理关键列的缺失值
    # 删除评分缺失的记录
    if 'imdb_score' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=['imdb_score'])
        after = len(df_clean)
        print(f"✅ 删除评分缺失记录: {before-after} 行")
    
    # 用中位数填充数值型列
    numeric_columns = ['duration', 'budget', 'gross', 'num_critic_for_reviews', 
                       'num_voted_users', 'num_user_for_reviews', 
                       'director_facebook_likes', 'cast_total_facebook_likes',
                       'movie_facebook_likes']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            missing_count = df_clean[col].isnull().sum()
            if missing_count == 0:
                print(f"✅ 已用中位数填充: {col}")
    
    # 用众数填充分类列
    categorical_columns = ['color', 'country', 'language', 'content_rating', 
                           'aspect_ratio', 'director_name']
    
    for col in categorical_columns:
        if col in df_clean.columns:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"✅ 已用众数填充: {col}")
    
    # 6. 创建新特征
    if 'title_year' in df_clean.columns:
        # 创建电影年龄特征
        current_year = datetime.now().year
        df_clean['movie_age'] = current_year - df_clean['title_year']
        
        # 创建年代特征
        df_clean['decade'] = (df_clean['title_year'] // 10) * 10
        print("✅ 已创建新特征: movie_age, decade")
    
    if 'gross' in df_clean.columns and 'budget' in df_clean.columns:
        # 创建投资回报率特征
        df_clean['roi'] = (df_clean['gross'] - df_clean['budget']) / df_clean['budget'].replace(0, np.nan)
        print("✅ 已创建新特征: roi (投资回报率)")
    
    # 7. 处理异常值
    if 'duration' in df_clean.columns:
        # 过滤时长在20-300分钟之间的电影
        before = len(df_clean)
        df_clean = df_clean[(df_clean['duration'] >= 20) & (df_clean['duration'] <= 300)]
        after = len(df_clean)
        print(f"✅ 过滤异常时长: 移除 {before-after} 行")
    
    print(f"\n🎉 数据清洗完成!")
    print(f"清洗后数据形状: {df_clean.shape}")
    
    return df_clean

def save_cleaned_data(df, filename="movies_cleaned.csv"):
    """保存清洗后的数据"""
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"💾 清洗后的数据已保存到: {output_path}")
    return output_path