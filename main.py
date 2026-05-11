# -*- coding: utf-8 -*-
"""
法律问答RAG助手
基于法律问答数据的RAG（检索增强生成）问答系统
使用Faiss向量数据库进行高效检索
"""

import os
import json
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Tuple
from openai import OpenAI
import pickle
import re
from sentence_transformers import SentenceTransformer

class LegalRAGAssistant:
    def __init__(self, data_path: str, api_key: str, base_url: str):
        """
        初始化法律RAG问答助手
        
        Args:
            data_path: 法律问答数据CSV文件路径
            api_key: 百度AI API密钥
            base_url: API基础URL
        """
        self.data_path = data_path
        self.faiss_index_path = "legal_faiss_index.bin"
        self.embeddings_path = "legal_embeddings.pkl"
        self.knowledge_db_path = "legal_knowledge.json"
        
        # 初始化AI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 初始化嵌入模型
        self.embedding_model = None
        self.faiss_index = None
        self.knowledge_base = []
        
        # 加载或构建知识库
        self._load_or_build_knowledge_base()
    
    def _get_embedding_model(self):
        """获取嵌入模型（延迟加载）"""
        if self.embedding_model is None:
            print("正在加载嵌入模型...")
            model_path = '/home/aistudio/models/paraphrase-multilingual-MiniLM-L12-v2'
            self.embedding_model = SentenceTransformer(model_path)
            print("嵌入模型加载完成")
        return self.embedding_model
    
    def _load_data(self) -> pd.DataFrame:
        """加载法律问答数据"""
        print("正在加载法律问答数据...")
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"成功加载 {len(df)} 条法律问答数据")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def _preprocess_data(self, df: pd.DataFrame) -> List[Dict]:
        """预处理数据"""
        print("正在预处理数据...")
        processed_data = []
        
        for _, row in df.iterrows():
            # 组合问题、标题和回答作为知识条目
            title = str(row.get('title', '')).strip()
            question = str(row.get('question', '')).strip()
            reply = str(row.get('reply', '')).strip()
            is_best = row.get('is_best', 0)
            
            # 跳过空数据
            if not reply or reply == 'nan' or reply == 'None':
                continue
            
            # 构建完整内容，确保不为空
            content = f"{title} {question} {reply}".strip()
            
            # 跳过内容为空的情况
            if not content or len(content) == 0:
                continue
            
            # 构建知识条目
            entry = {
                'title': title,
                'question': question,
                'reply': reply,
                'is_best': is_best,
                'content': content,
                'full_text': f"问题：{question}\n回答：{reply}"
            }
            processed_data.append(entry)
        
        print(f"预处理完成，有效条目：{len(processed_data)}")
        return processed_data
    
    def _build_faiss_index(self):
        """构建Faiss向量索引"""
        print("正在构建Faiss向量索引...")
        
        # 获取嵌入模型
        embedding_model = self._get_embedding_model()
        
        # 生成所有文本的嵌入向量，过滤空文本
        texts = []
        valid_entries = []
        for entry in self.knowledge_base:
            content = entry.get('content', '').strip()
            if content and len(content) > 0:  # 确保内容不为空
                texts.append(content)
                valid_entries.append(entry)
        
        # 更新知识库为有效条目
        self.knowledge_base = valid_entries
        
        print(f"正在生成 {len(texts)} 个文本的嵌入向量...")
        
        if len(texts) == 0:
            print("警告：没有有效的文本用于生成嵌入向量")
            return
        
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
        # 创建Faiss索引
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        
        # 归一化向量（用于余弦相似度）
        faiss.normalize_L2(embeddings)
        
        # 添加向量到索引
        self.faiss_index.add(embeddings.astype('float32'))
        
        # 保存索引和嵌入向量
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Faiss索引构建完成，维度：{dimension}")
    
    def _load_or_build_knowledge_base(self):
        """加载或构建知识库"""
        # 检查是否存在已保存的知识库和索引
        if (os.path.exists(self.knowledge_db_path) and 
            os.path.exists(self.faiss_index_path) and 
            os.path.exists(self.embeddings_path)):
            try:
                print("加载已有知识库和Faiss索引...")
                
                # 加载知识库
                with open(self.knowledge_db_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                
                # 加载Faiss索引
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                
                print(f"知识库加载完成，共{len(self.knowledge_base)}个条目")
                return
            except Exception as e:
                print(f"加载知识库失败：{e}，重新构建...")
        
        # 构建新的知识库
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """构建知识库"""
        print("正在构建知识库...")
        
        # 加载数据
        df = self._load_data()
        if df.empty:
            print("数据加载失败，无法构建知识库")
            return
        
        # 预处理数据
        self.knowledge_base = self._preprocess_data(df)
        
        # 保存知识库
        with open(self.knowledge_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        
        # 构建Faiss索引
        self._build_faiss_index()
        
        print(f"知识库构建完成，共{len(self.knowledge_base)}个条目")
    
    def _retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关文档"""
        if self.faiss_index is None or not self.knowledge_base:
            return []
        
        # 验证查询字符串
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            print("警告：查询字符串为空或无效")
            return []
        
        # 获取嵌入模型
        embedding_model = self._get_embedding_model()
        
        # 生成查询向量
        query_embedding = embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索最相似的文档
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # 获取相关文档
        relevant_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.knowledge_base):
                doc = self.knowledge_base[idx].copy()
                doc['similarity_score'] = float(score)
                doc['rank'] = i + 1
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def answer_question(self, question: str) -> str:
        """回答法律问题"""
        # 检索相关文档
        relevant_docs = self._retrieve_relevant_documents(question, top_k=3)
        
        if not relevant_docs:
            return "抱歉，没有找到相关的法律信息。"
        
        # 构建上下文
        context = "相关法律知识：\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc['full_text']}\n\n"
        
        # 构建提示词
        prompt = f"""
你是一个专业的法律顾问。请根据以下相关法律知识，为用户的法律问题提供准确、专业的回答。

{context}

用户问题：{question}

请提供：
1. 直接回答用户的问题
2. 引用相关的法律条文或规定
3. 给出具体的建议或解决方案
4. 如果涉及具体案件，建议咨询专业律师

回答要求：
- 准确、专业、易懂
- 基于提供的法律知识
- 避免给出具体的法律建议，仅提供参考信息
"""
        
        # 调用AI API
        try:
            response = self.client.chat.completions.create(
                model="ernie-4.5-21b-a3b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
                top_p=0.8,
                extra_body={
                    "penalty_score": 1
                }
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"回答生成失败：{str(e)}"
    
    def interactive_qa(self):
        """交互式问答"""
        print("=== 法律问答RAG助手 ===")
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'help' 查看帮助信息")
        print()
        
        while True:
            try:
                question = input("请输入您的法律问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if question.lower() == 'help':
                    print("""
帮助信息：
- 输入法律相关问题，获取专业回答
- 支持各种法律领域的咨询
- 输入 'quit' 或 'exit' 退出程序
- 注意：本系统仅供参考，具体法律问题请咨询专业律师
                    """)
                    continue
                
                if not question:
                    print("请输入有效问题")
                    continue
                
                print("\n正在检索相关法律信息...")
                answer = self.answer_question(question)
                print(f"\n回答：\n{answer}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"发生错误：{e}")

def main():
    """主函数"""
    # 配置参数
    data_path = r"D:\workplace\yt\法律问答数据.csv"
    api_key = "7b97b5e65d1248169aab2d56f67d2b0fbcb146a2"
    base_url = "https://aistudio.baidu.com/llm/lmapi/v3"
    
    # 创建法律RAG助手
    assistant = LegalRAGAssistant(data_path, api_key, base_url)
    
    # 启动交互式问答
    assistant.interactive_qa()

if __name__ == "__main__":
    main()
