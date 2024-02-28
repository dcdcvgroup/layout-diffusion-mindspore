# Setup Dataset 
请参考 [Layout2Im](https://github.com/zhaobozb/layout2im), [LostGAN](https://github.com/WillSuen/LostGANs), 使用 [Sg2Im](https://github.com/google/sg2im/tree/master/scripts)仓库的数据预处理代码。

### COCO-stuff 2017 
```bash
  bash bash/download_coco.sh
```

  <details><summary>COCO 2017文件结构</summary>

   ```
   ├── annotations
   │    └── deprecated-challenge2017
   │         └── train-ids.txt
   │         └── val-ids.txt
   │    └── instances_train2017.json
   │    └── instances_val2017.json
   │    └── stuff_train2017.json
   │    └── stuff_val2017.json
   │    └── ...
   ├── images
   │    └── train2017
   │         └── 000000000872.jpg
   │         └── ... 
   │   └── val2017
   │         └── 000000000321.jpg
   │         └── ... 
   ```

   </details>

### Visual Genome 
```bash
  # 运行以下脚本下载并解压 Visual Genome 数据集的相关部分。
  # 这将创建目录 datasets/vg 并将大约 15 GB 的数据下载到该目录； 解压后大约需要30GB的磁盘空间。
  bash bash/download_vg.sh
  
  # 下载Visual Genome数据集后，我们需要对其进行预处理。 这会将数据拆分为训练/验证/测试拆分，将所有场景图合并到 HDF5 文件中，并应用多种启发式方法来清理数据。 特别是，我们忽略太小的图像，只考虑在训练集中出现一定次数的对象和属性类别； 我们还忽略太小的对象，并设置每个图像中出现的对象和关系的数量的最小值和最大值。
  # 这将在目录 datasets/vg 中创建文件 train.h5、val.h5、test.h5 和 vocab.json。
  python scripts/preprocess_vg.py
```

   <details><summary>Visual Genome文件结构</summary>

   ```
   ├── VG_100K
   │   └── captions_val2017.json
   │   └── ...
   └── objects.json
   └── train.json
   └── ...
   ```

   </details>