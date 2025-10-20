import hashlib
import time

class Block:
    def __init__(self, index, timestamp, data, previous_hash, kp, acc, loss):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
        self.kp = kp
        self.acc = acc
        self.loss = loss

    def calculate_hash(self):
        """
        计算当前区块的哈希值
        """
        block_string = str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        """
        创建创世区块
        """
        return Block(0, time.time(), "Genesis Block", "0", 0, 0, 10)

    def get_latest_block(self):
        """
        获取链中的最新区块
        """
        return self.chain[-1]

    def add_block(self, new_block):
        """
        添加新的区块到链中
        """
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)


# 验证区块链
def validate(blockchain):
    bool = True
    # 上一个区块
    previous_index = 0
    previous_hash = 0
    for block in blockchain.chain:
        index = block.index
        hash = block.hash
        if (index > 0):
            # 如果index是衔接的
            if previous_index == index - 1:
                pass
            else:
                bool = False
            # 如果上一个区块的当前hash和当前区块的上一个hash值能对上
            if (previous_hash == block.previous_hash):
                pass
            else:
                bool = False

            if bool:
                # 把当前的变为上一个
                previous_index = index
                previous_hash = hash
        if index == 0:
            previous_index = index
            previous_hash = hash
            pass
    return bool