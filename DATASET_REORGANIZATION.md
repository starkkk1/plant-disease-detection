# Tong Quan Dataset

## 1. Muc tieu to chuc dataset

Dataset trong project nay duoc to chuc lai de phuc vu bai toan **phan loai benh la ca chua**. Muc tieu cua cach to chuc moi la:

- su dung toan bo du lieu tomato trong PlantVillage lam nguon huan luyen chinh
- tach ro du lieu phuc vu huan luyen va du lieu phuc vu danh gia
- khong tron cac bo test co nhan khac nhau vao cung mot thu muc
- giu lai cac bo du lieu ngoai phan phoi de kiem tra kha nang tong quat hoa cua mo hinh

Noi ngan gon, dataset moi duoc chia thanh 3 phan:

- `train`: du lieu dung de huan luyen
- `val`: du lieu dung de theo doi, tinh toan metric trong qua trinh huan luyen va chon mo hinh
- `eval`: cac bo benchmark rieng dung de danh gia sau khi huan luyen

## 2. Nguon du lieu duoc su dung

Dataset moi duoc tao tu cac nguon trong `data+source` va mot phan du lieu da xu ly san trong project.

### 2.1. Nguon chinh: PlantVillage tomato-only

Nguon:

```text
data+source/PlantVillage-Dataset/raw/color/
```

Sau khi ban chinh sua, thu muc nay chi con cac lop cua **tomato**. Day la nguon du lieu chinh de tao ra `train` va `val`.

Tong cong co **10 lop**:

| STT | Lop | So anh goc |
|---|---|---:|
| 1 | Tomato___Bacterial_spot | 2127 |
| 2 | Tomato___Early_blight | 1000 |
| 3 | Tomato___healthy | 1591 |
| 4 | Tomato___Late_blight | 1909 |
| 5 | Tomato___Leaf_Mold | 952 |
| 6 | Tomato___Septoria_leaf_spot | 1771 |
| 7 | Tomato___Spider_mites Two-spotted_spider_mite | 1676 |
| 8 | Tomato___Target_Spot | 1404 |
| 9 | Tomato___Tomato_mosaic_virus | 373 |
| 10 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5357 |

Tong so anh goc tu PlantVillage tomato:

- **18,160 anh**

Luu y:

- Trong project, ten lop `Tomato___Spider_mites Two-spotted_spider_mite` duoc chuan hoa thanh `Tomato___Spider_mites_Two_spotted_spider_mite` de tranh loi ten thu muc va de dong nhat voi code.

### 2.2. Bo test 3 lop da co tu truoc

Nguon:

```text
plant-disease-detection/data/processed_custom/test/
```

Bo nay duoc giu lai de danh gia mo hinh tren bai toan 3 lop cu:

- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___healthy`

Bo nay khong duoc dung de train trong cau truc moi. No duoc dua vao `eval/three_class_external`.

### 2.3. Bo Bangladesh binary benchmark

Nguon:

```text
data+source/Tomato_Leaves/
```

Bo nay co dang annotation kieu object detection voi file nhan YOLO. Script se crop la tu bounding box va tao thanh bo danh gia nhi phan:

- `Tomato___healthy`
- `Tomato___diseased`

Bo nay khong phai bai toan 10 lop. No duoc dung de kiem tra mo hinh co phan biet duoc la lanh va la benh tren nguon du lieu khac hay khong.

### 2.4. Bo multi-leaf benchmark

Nguon:

```text
data+source/Tomato-Village/Variant-c(Object Detection)/
```

Bo nay cung co dang object detection. Script se crop tu bounding box va dua vao bo danh gia `multileaf_late_blight`.

Bo nay chu yeu huu ich de danh gia tinh on dinh cua mo hinh tren anh phuc tap hon, nhieu la hon, dac biet voi:

- `Tomato___Late_blight`

## 3. Cau truc dataset moi

Dataset moi duoc tao tai:

```text
plant-disease-detection/data/plantvillage_tomato_full/
```

Cau truc tong quat:

```text
plantvillage_tomato_full/
├── train/
├── val/
├── eval/
│   ├── three_class_external/
│   ├── bangladesh_binary/
│   └── multileaf_late_blight/
└── metadata/
    ├── class_to_idx.json
    └── dataset_summary.json
```

Y nghia tung thu muc:

- `train/`: du lieu huan luyen chinh tu PlantVillage tomato
- `val/`: du lieu validation tu cung nguon PlantVillage tomato
- `eval/three_class_external/`: bo test 3 lop cu
- `eval/bangladesh_binary/`: bo benchmark healthy vs diseased
- `eval/multileaf_late_blight/`: bo benchmark anh nhieu la, chu yeu cho Late blight
- `metadata/`: chua file mapping nhan va thong ke du lieu

## 4. Cach chia du lieu

Toan bo du lieu PlantVillage tomato duoc chia theo ty le:

- `train`: 80%
- `val`: 20%

Seed su dung:

- `42`

Cach chia nay duoc thuc hien rieng cho tung lop, nen van giu duoc phan bo lop hop ly giua `train` va `val`.

Trong thiet ke nay:

- khong tao mot bo `test` noi phan phoi moi tu PlantVillage
- thay vao do, su dung cac bo trong `eval/` de danh gia sau khi huan luyen

Ly do la vi project nay uu tien:

- dung nhieu du lieu PlantVillage nhat co the cho huan luyen
- van giu `val` de theo doi va chon mo hinh
- danh gia bang cac benchmark tach rieng de tranh nham lan giua cac bai toan

## 5. So lieu chi tiet sau khi reorganize

Sau khi chay script `prepare_plantvillage_tomato_full.py`, dataset moi co so lieu nhu sau.

### 5.1. Train set

| Lop | So anh |
|---|---:|
| Tomato___Bacterial_spot | 1701 |
| Tomato___Early_blight | 800 |
| Tomato___Late_blight | 1527 |
| Tomato___Leaf_Mold | 761 |
| Tomato___Septoria_leaf_spot | 1416 |
| Tomato___Spider_mites_Two_spotted_spider_mite | 1340 |
| Tomato___Target_Spot | 1123 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 4285 |
| Tomato___Tomato_mosaic_virus | 298 |
| Tomato___healthy | 1272 |

Tong so anh train:

- **14,523 anh**

### 5.2. Validation set

| Lop | So anh |
|---|---:|
| Tomato___Bacterial_spot | 426 |
| Tomato___Early_blight | 200 |
| Tomato___Late_blight | 382 |
| Tomato___Leaf_Mold | 191 |
| Tomato___Septoria_leaf_spot | 355 |
| Tomato___Spider_mites_Two_spotted_spider_mite | 336 |
| Tomato___Target_Spot | 281 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1072 |
| Tomato___Tomato_mosaic_virus | 75 |
| Tomato___healthy | 319 |

Tong so anh val:

- **3,637 anh**

### 5.3. Evaluation set: three_class_external

| Lop | So anh |
|---|---:|
| Tomato___Early_blight | 50 |
| Tomato___Late_blight | 92 |
| Tomato___healthy | 22 |

Tong so anh:

- **164 anh**

Y nghia:

- dung de kiem tra mo hinh 10 lop co con hoat dong tot tren bai toan 3 lop cu hay khong

### 5.4. Evaluation set: bangladesh_binary

| Lop | So anh |
|---|---:|
| Tomato___diseased | 607 |
| Tomato___healthy | 428 |

Tong so anh:

- **1,035 anh**

Y nghia:

- dung de kiem tra mo hinh tren nguon khac
- phu hop cho goc nhin healthy vs diseased
- khong phu hop de gop chung voi bai toan 10 lop

### 5.5. Evaluation set: multileaf_late_blight

| Lop | So anh |
|---|---:|
| Tomato___Late_blight | 3938 |

Tong so anh:

- **3,938 anh**

Y nghia:

- dung de danh gia do ben cua mo hinh tren anh co boi canh phuc tap hon
- bo nay chu yeu phuc vu kiem tra rieng doi voi `Late_blight`

## 6. Vai tro cua tung phan trong quy trinh huan luyen

### Train

`train/` duoc dung de:

- huan luyen mo hinh
- hoc cac dac trung cua 10 lop benh la ca chua

### Validation

`val/` duoc dung de:

- theo doi loss va metric trong qua trinh train
- chon checkpoint tot nhat
- kiem tra overfitting
- tinh toan cac metric nhu accuracy, macro F1, recall theo lop

### Evaluation

`eval/` duoc dung de:

- danh gia sau khi da chot mo hinh
- khong tham gia vao qua trinh update tham so
- kiem tra kha nang tong quat hoa tren cac nguon du lieu khac nhau

## 7. Diem manh cua cach to chuc nay

- Su dung duoc toan bo 10 lop tomato cua PlantVillage thay vi chi 3 lop.
- Co `train` va `val` ro rang de train va tuning mo hinh dung cach.
- Khong tron benchmark ngoai vao train, giup ket qua danh gia minh bach hon.
- Co nhieu bo `eval` khac nhau de nhin ro tung khia canh tong quat hoa.
- Phu hop voi bai toan nghien cuu va viet bao cao vi co the trinh bay ket qua theo tung benchmark.

## 8. Diem yeu va gioi han

- Nguon train chinh van la PlantVillage, ma PlantVillage la du lieu kha sach, nen ket qua tren `val` co the lac quan hon so voi thuc te.
- `three_class_external` chi gom 3 lop, khong phan anh day du bai toan 10 lop.
- `bangladesh_binary` la bai toan nhi phan, khac ban chat voi bai toan multiclass 10 lop.
- `multileaf_late_blight` chi tap trung vao 1 lop, nen khong the xem la test set can bang.
- Vi vay, khong nen bao cao chi mot con so "test accuracy" duy nhat cho toan bo project.

## 9. Cach bao cao ket qua de hop ly

Nen bao cao ket qua theo thu tu sau:

1. Ket qua tren `val` cho bai toan 10 lop PlantVillage tomato
2. Ket qua tren `eval/three_class_external` cho 3 lop trung nhau
3. Ket qua tren `eval/bangladesh_binary` theo goc nhin healthy vs diseased
4. Ket qua tren `eval/multileaf_late_blight` de nhan xet do ben va kha nang ung pho voi boi canh phuc tap

Neu viet vao bao cao, ban co the dien dat ngan gon nhu sau:

> Dataset cua he thong duoc xay dung tu nguon PlantVillage tomato-only lam tap phat trien chinh, gom 10 lop benh la ca chua va duoc chia thanh train/validation. Ben canh do, cac bo du lieu ngoai nhu three-class external, Bangladesh binary va multi-leaf benchmark duoc giu tach rieng trong thu muc evaluation de danh gia kha nang tong quat hoa cua mo hinh tren nhieu dieu kien khac nhau.

## 10. Ket luan

Cach to chuc dataset hien tai hop ly cho muc tieu cua project vi:

- du lieu huan luyen chinh ro rang
- so lop day du hon so voi ban 3 lop truoc day
- cac bo danh gia duoc tach rieng theo dung ban chat cua tung bai toan

Tuy nhien, de tranh danh gia qua lac quan, can luon nho rang:

- ket qua tren `val` chu yeu phan anh kha nang hoc tren du lieu PlantVillage
- ket qua tren `eval` moi cho thay muc do tong quat hoa khi gap nguon du lieu khac
