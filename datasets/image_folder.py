from typing import Optional, Callable, Any

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class CustomImageFolder(ImageFolder):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.overfit = False
        self._fixed_batch_size = None

    def set_overfit_mode(self, batch_size):
        self.overfit = True
        self._fixed_batch_size = batch_size

    def unset_overfit_mode(self):
        self.overfit = False

    def __len__(self) -> int:
        return len(self.samples) if not self.overfit else self._fixed_batch_size
