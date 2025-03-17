import os
import lmdb
import cv2
import numpy as np
import pickle
from typing import Union, Tuple, Any

class LMDBHandler:
    def __init__(self, db_path: str, map_size: int = 1099511627776):
        """初始化LMDB数据库
        
        Args:
            db_path: LMDB数据库路径
            map_size: 数据库大小限制，默认为1TB
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)
    
    def write_tuple(self, key: Union[str, bytes], data: Tuple) -> None:
        """将元组写入LMDB
        
        Args:
            key: 数据的唯一标识符
            data: 要存储的元组数据
        """
        if isinstance(key, str):
            key = key.encode()
        
        # 使用pickle序列化元组
        serialized_data = pickle.dumps(data)
        
        # 写入数据库
        with self.env.begin(write=True) as txn:
            txn.put(key, serialized_data)
    
    def read_tuple(self, key: Union[str, bytes]) -> Tuple:
        """从LMDB读取元组
        
        Args:
            key: 数据的唯一标识符
            
        Returns:
            存储的元组数据
        """
        if isinstance(key, str):
            key = key.encode()
            
        # 从数据库读取
        with self.env.begin() as txn:
            serialized_data = txn.get(key)
            if serialized_data is None:
                raise KeyError(f"找不到键值为{key}的数据")
            
            # 反序列化数据
            return pickle.loads(serialized_data)
    
    def write_batch_tuples(self, items: list[Tuple[Union[str, bytes], Tuple]]) -> None:
        """批量写入元组
        
        Args:
            items: 包含(key, tuple_data)的列表
        """
        with self.env.begin(write=True) as txn:
            for key, data in items:
                if isinstance(key, str):
                    key = key.encode()
                serialized_data = pickle.dumps(data)
                txn.put(key, serialized_data)
    
    def write_image(self, key: Union[str, bytes], img: np.ndarray) -> None:
        """将图片写入LMDB
        
        Args:
            key: 图片的唯一标识符
            img: OpenCV格式的图片数组(BGR)
        """
        if isinstance(key, str):
            key = key.encode()
            
        # 将图片编码为jpg格式
        success, encoded_img = cv2.imencode('.jpg', img)
        if not success:
            raise ValueError("图片编码失败")
            
        # 写入数据库
        with self.env.begin(write=True) as txn:
            txn.put(key, encoded_img.tobytes())
    
    def read_image(self, key: Union[str, bytes]) -> np.ndarray:
        """从LMDB读取图片
        
        Args:
            key: 图片的唯一标识符
            
        Returns:
            OpenCV格式的图片数组(BGR)
        """
        if isinstance(key, str):
            key = key.encode()
            
        # 从数据库读取
        with self.env.begin() as txn:
            img_bytes = txn.get(key)
            if img_bytes is None:
                raise KeyError(f"找不到键值为{key}的图片")
                
            # 解码图片
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    
    def exists(self, key: Union[str, bytes]) -> bool:
        """检查键是否存在
        
        Args:
            key: 要检查的键
            
        Returns:
            bool: 键是否存在
        """
        if isinstance(key, str):
            key = key.encode()
        with self.env.begin() as txn:
            return txn.get(key) is not None
    
    def close(self):
        """关闭数据库连接"""
        self.env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 