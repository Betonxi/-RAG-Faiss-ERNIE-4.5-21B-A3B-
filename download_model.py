#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地下载Sentence Transformers模型脚本
运行此脚本下载模型，然后上传到云端使用
"""

import os
import sys
from sentence_transformers import SentenceTransformer

def download_model():
    """下载指定的模型"""
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    print(f"正在下载模型: {model_name}")
    print("这可能需要几分钟时间，请耐心等待...")
    
    try:
        # 下载模型
        model = SentenceTransformer(model_name)
        
        # 获取模型路径
        model_path = model._modules['0'].auto_model.config._name_or_path
        print(f"\n✓ 模型下载成功！")
        print(f"模型路径: {model_path}")
        
        # 测试模型
        print("\n正在测试模型...")
        test_text = "这是一个测试文本"
        embedding = model.encode([test_text])
        print(f"✓ 模型测试成功，向量维度: {embedding.shape[1]}")
        
        # 显示模型文件
        print(f"\n模型文件列表:")
        for file in os.listdir(model_path):
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  {file}: {size:.2f} MB")
        
        print(f"\n下一步:")
        print(f"1. 找到模型文件夹: {model_path}")
        print(f"2. 将整个文件夹压缩为zip文件")
        print(f"3. 上传到BML CodeLab")
        print(f"4. 在云端解压并使用")
        
        return model_path
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 尝试使用VPN")
        print("3. 使用备用模型")
        return None

def download_backup_model():
    """下载备用模型"""
    backup_models = [
        'paraphrase-multilingual-MiniLM-L6-v2',
        'all-MiniLM-L6-v2'
    ]
    
    for model_name in backup_models:
        print(f"\n尝试下载备用模型: {model_name}")
        try:
            model = SentenceTransformer(model_name)
            model_path = model._modules['0'].auto_model.config._name_or_path
            print(f"✓ 备用模型下载成功: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ 备用模型下载失败: {e}")
            continue
    
    return None

if __name__ == "__main__":
    print("=== Sentence Transformers 模型下载工具 ===")
    print("此工具将下载模型到本地，然后您可以上传到云端使用")
    print()
    
    # 尝试下载主模型
    model_path = download_model()
    
    # 如果主模型失败，尝试备用模型
    if model_path is None:
        print("\n主模型下载失败，尝试备用模型...")
        model_path = download_backup_model()
    
    if model_path:
        print(f"\n🎉 模型下载完成！")
        print(f"请将模型文件夹上传到云端: {model_path}")
    else:
        print("\n❌ 所有模型下载都失败了")
        print("请检查网络连接或联系技术支持")
