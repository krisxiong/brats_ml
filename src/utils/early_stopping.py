"""
早停机制模块
"""

from typing import Optional


class ImprovedEarlyStopping:
    """改进的早停机制"""

    def __init__(self, patience: int = 20, min_delta: float = 0.0001,
                 verbose: bool = True, mode: str = 'max'):
        """
        初始化早停机制

        Args:
            patience: 容忍epoch数
            min_delta: 最小改进阈值
            verbose: 是否打印信息
            mode: 'max'（指标越大越好）或'min'（指标越小越好）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch: int, score: float) -> bool:
        """
        检查是否早停

        Args:
            epoch: 当前epoch
            score: 当前指标分数

        Returns:
            是否触发早停
        """
        if self.mode == 'max':
            score = score
        else:
            score = -score  # 转换为最大值问题

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"  [早停] 初始最佳: {score:.4f}")

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [早停] 无改进 {self.counter}/{self.patience} "
                      f"(当前: {score:.4f}, 最佳: {self.best_score:.4f})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n  [早停] 触发！")
                    print(f"    最佳: {self.best_score:.4f} (Epoch {self.best_epoch})")
                    print(f"    当前: {score:.4f} (Epoch {epoch})")

        else:
            improvement = score - self.best_score
            if self.verbose:
                print(f"  [早停] 改进! +{improvement:.4f} "
                      f"(从 {self.best_score:.4f} → {score:.4f})")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0  # 重置计数器

        return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class EarlyStopping:
    """基础早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.mode == 'max':
            score = score
        else:
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop