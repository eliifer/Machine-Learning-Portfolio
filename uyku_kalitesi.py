import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score




df = pd.read_csv(r"C:\Users\lenovo\Downloads\archive (10)\Sleep_Efficiency.csv")
df.head(20)

df.shape
df.columns
df.isnull().sum()
df.describe().T
df.info()

df['Bedtime'] = pd.to_datetime(df['Bedtime'])
df['Wakeup time'] = pd.to_datetime(df['Wakeup time'])

df['Bedtime_Hour'] = df['Bedtime'].dt.hour
df['Wakeup_Hour'] = df['Wakeup time'].dt.hour

df.drop(['ID', 'Bedtime', 'Wakeup time'], axis=1, inplace=True)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Smoking status'] = le.fit_transform(df['Smoking status'])


######################################################################
cat_cols =[col for col in df.columns if df[col].nunique() <= 2]
num_cols = [col for col in df.columns if col not in cat_cols and col != 'Sleep efficiency']


def cat_summary_with_target(dataframe, target, categorical_col):
    print(f"--- {categorical_col} bazlı {target} Ortalamaları ---")
    summary = dataframe.groupby(categorical_col).agg({target: ["mean", "count"]})
    summary.columns = ["Hedef_Ortalama", "Gözlem_Sayısı"]
    print(summary, end="\n\n")

for col in cat_cols:
    cat_summary_with_target(df, "Sleep efficiency", col)


def num_summary_plots(dataframe, numerical_col, target):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(dataframe[numerical_col], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"{numerical_col} Dağılımı")

    sns.regplot(x=numerical_col, y=target, data=dataframe, ax=axes[1],
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    axes[1].set_title(f"{numerical_col} vs {target}")

    plt.tight_layout()
    plt.show()


for col in num_cols:
    num_summary_plots(df, col, "Sleep efficiency")


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.violinplot(x='Gender', y='Sleep efficiency', data=df, palette="muted", inner="quartile")
plt.title('Cinsiyete Göre Uyku Verimliliği Dağılımı', fontsize=14)


plt.subplot(1, 2, 2)
sns.violinplot(x='Smoking status', y='Sleep efficiency', data=df, palette="pastel", inner="quartile")
plt.title('Sigara Kullanımına Göre Uyku Verimliliği Dağılımı', fontsize=14)

plt.tight_layout()
plt.show()


cols_for_pairplot = ['Deep sleep percentage', 'Alcohol consumption', 'Exercise frequency', 'Sleep efficiency']

sns.pairplot(df[cols_for_pairplot],
             diag_kind='kde',   # Köşegenlerde yoğunluk grafiği (KDE) olsun
             plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
             palette='viridis')

plt.suptitle('Kritik Değişkenlerin Etkileşim Matrisi', y=1.02, fontsize=16)
plt.show()



######################################################
#EKSİK DEĞER ANALİZİ BÖLÜMÜ
#######################################################


df.head(20)
df.isnull().sum()
df.notnull().sum()
df.isnull().sum().sum()
df[df.isnull().any(axis=1)]
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)


msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()




def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Sleep efficiency", na_cols)

######################################################################################
#CORR MATRİS
######################################################################################

plt.figure(figsize=(14, 10))

corr_matrix = df.corr()

sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            vmin=-1, vmax=1)

plt.title('Sleep Efficiency Veri Seti - Korelasyon Isı Haritası', fontsize=16)

plt.tight_layout()
plt.show()

########################################################################################
#EKSİK DEĞER DOLDURMA BÖLÜMÜ - KNN IMPUTER
#####################################################################################


scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

knn_imputer = KNNImputer(n_neighbors=5)
df_imputed_scaled = pd.DataFrame(knn_imputer.fit_transform(df_scaled), columns=df.columns)

df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df.columns)

print("\nKNN Imputation Sonrası Eksik Değerler:")
print(df_imputed.isnull().sum().sum())

df.head(20)
df_imputed.head(20)


num_cols = [col for col in df_imputed.columns if (df_imputed[col].nunique() > 2) and (col != "Sleep efficiency")]
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(4, 3, i+1)
    sns.boxplot(y=df_imputed[col], color='skyblue')
    plt.title(col)

plt.tight_layout()
plt.show()



##############################################################################
#AYKIRI GÖZLEM ANALİZİ
#############################################################################


# LOF modelini kuruyoruz

#
# lof = LocalOutlierFactor(n_neighbors=20)
#
# # Sadece sürekli sayısal değişkenleri vererek tahmin yaptırıyoruz
# # LOF normal değerlere 1, aykırı değerlere -1 döndürür
# outlier_predictions = lof.fit_predict(df_imputed_scaled[num_cols])
# # Aykırı olanları True yapan bir maske oluşturuyoruz
# outliers_mask = outlier_predictions == -1
#
# print(f"Toplam gözlem sayısı: {len(df_imputed)}")
# print(f"LOF tarafından tespit edilen aykırı değer sayısı: {outliers_mask.sum()}")
#
# # Aykırı değerleri veri setinden atarak temiz bir dataframe elde ediyoruz (~ işareti tersini alır)
# df_clean = df_imputed[~outliers_mask].reset_index(drop=True)
# print(f"Aykırı değerler atıldıktan sonra temiz veri seti boyutu: {df_clean.shape}")
#
# df_outliers = df_imputed[outliers_mask]
#
# print(f"Tespit edilen aykırı değer sayısı: {len(df_outliers)}")
df_OUTLİER =df_imputed.copy()

print("--- IQR (%10 - %90) Aykırı Değer Baskılama İşlemi ---")

for col in num_cols:
    Q1 = df_imputed[col].quantile(0.10)
    Q3 = df_imputed[col].quantile(0.90)
    IQR = Q3 - Q1

    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR

    aykiri_sayisi = ((df_imputed[col] < alt_sinir) | (df_imputed[col] > ust_sinir)).sum()

    if aykiri_sayisi > 0:
        print(f"{col}: {aykiri_sayisi} adet aykırı değer sınırlara çekildi.")

    df_imputed[col] = np.where(df_imputed[col] < alt_sinir, alt_sinir, df_imputed[col])
    df_imputed[col] = np.where(df_imputed[col] > ust_sinir, ust_sinir, df_imputed[col])

print("\nİşlem Tamamlandı!")
print(f"Temiz veri seti boyutu: {df_imputed.shape}")




##########################################################
# corelasyon
###########################################################



plt.figure(figsize=(14, 10))

corr_matrix = df_imputed.corr()

sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            vmin=-1, vmax=1)

plt.title('Sleep Efficiency Veri Seti - Korelasyon Isı Haritası', fontsize=16)

plt.tight_layout()
plt.show()


corr_matrix = df_imputed.corr().abs()

high_corr_var=np.where(corr_matrix>0.85)
high_corr_pairs = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]

print("Yüksek Korelasyonlu Çiftler:", high_corr_pairs)

df_final= df_imputed.drop('Light sleep percentage', axis=1)



#################33###########################################33######################################
#model kurma kısmı
#######################################################################################################


X = df_final.drop('Sleep efficiency', axis=1)
y = df_final['Sleep efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


from sklearn.linear_model import LinearRegression, LassoCV, RANSACRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. OLS
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)
ols_pred = ols_model.predict(X_test_scaled)

# 2. Lasso Modeli (LassoCV ile en iyi alpha değerini otomatik buluyoruz)
lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)
print(f"LassoCV tarafından bulunan en iyi Alpha değeri: {lasso_model.alpha_:.4f}")

# 3. RANSAC Modeli
ransac_model = RANSACRegressor(random_state=42)
ransac_model.fit(X_train_scaled, y_train)
ransac_pred = ransac_model.predict(X_test_scaled)
################################################################################33333







fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- GÖRSEL 1: Lasso Katsayıları (Özellik Seçimi) ---
# Katsayıları bir DataFrame'e alıp büyüklüğe göre sıralayalım
lasso_coefs = pd.DataFrame({'Özellik': X.columns, 'Katsayı': lasso_model.coef_})
lasso_coefs = lasso_coefs.sort_values(by='Katsayı', ascending=False)

sns.barplot(x='Katsayı', y='Özellik', data=lasso_coefs, ax=axes[0], palette='coolwarm')
axes[0].set_title('Lasso Regresyonu Katsayıları (Özellik Seçimi)', fontsize=14)
axes[0].axvline(0, color='black', linewidth=1)

# --- GÖRSEL 2: RANSAC Inlier/Outlier Yönetimi ---

feature_idx = X.columns.get_loc('Sleep duration')
x_plot = X_train_scaled.iloc[:, feature_idx]

inlier_mask = ransac_model.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

axes[1].scatter(x_plot[inlier_mask], y_train[inlier_mask], color='green', marker='o', label='Inliers (Geçerli Veri)', alpha=0.6)
axes[1].scatter(x_plot[outlier_mask], y_train[outlier_mask], color='red', marker='x', label='Outliers (Aykırı Veri)', s=70)
axes[1].set_title("RANSAC'ın Aykırı Değer Yönetimi (Eğitim Seti Üzerinde)", fontsize=14)
axes[1].set_xlabel("Standartlaştırılmış Uyku Süresi")
axes[1].set_ylabel("Uyku Verimliliği (Hedef)")
axes[1].legend()

plt.tight_layout()
plt.show()

# Metrikleri hesaplayan yardımcı bir fonksiyon
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [model_name, mse, mae, r2]


##################################################################################################

results = []
results.append(evaluate_model(y_test, ols_pred, 'OLS (Linear Regression)'))
results.append(evaluate_model(y_test, lasso_pred, 'Lasso Regression (CV)'))
results.append(evaluate_model(y_test, ransac_pred, 'RANSAC Regression'))

results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'MAE', 'R2_Score'])
results_df.set_index('Model', inplace=True)

print("\n--- Model Performans Karşılaştırması ---")
print(results_df.round(4))


df['Sleep efficiency'].mean()
###############################################################################################3

##############################################
#losso hangi özellkleri eledii?

# Lasso'nun katsayısını TAMAMEN 0 yaptığı özellikleri bulalım:
tamamen_elenenler = lasso_coefs[lasso_coefs['Katsayı'] == 0]['Özellik'].tolist()
print("Lasso tarafından TAMAMEN elenen özellikler (Katsayı = 0):", tamamen_elenenler)

# Lasso'nun katsayısını çok küçülterek ETKİSİZ HALE getirdiği özellikleri bulalım
etkisiz_ozellikler = lasso_coefs[abs(lasso_coefs['Katsayı']) < 0.001]['Özellik'].tolist()
print("Lasso tarafından BASKILANAN / ETKİSİZLEŞTİRİLEN özellikler:", etkisiz_ozellikler)


#################################################################################################################################333



print("\n--- STANDARTLAŞTIRMANIN (SCALING) ETKİSİ DENEYİ ---")

# 1. HAM (Ölçeklenmemiş) Veri ile Lasso Modeli Kurulumu
lasso_unscaled = LassoCV(cv=5, random_state=42)
lasso_unscaled.fit(X_train, y_train)
unscaled_pred = lasso_unscaled.predict(X_test)
unscaled_r2 = r2_score(y_test, unscaled_pred)



compare_coefs = pd.DataFrame({
    'Özellik': X.columns,
    'Ham Veri Katsayısı (Unscaled)': lasso_unscaled.coef_,
    'Ölçeklenmiş Veri Katsayısı (Scaled)': lasso_model.coef_
})

compare_coefs['Mutlak_Scaled'] = compare_coefs['Ölçeklenmiş Veri Katsayısı (Scaled)'].abs()
compare_coefs = compare_coefs.sort_values(by='Mutlak_Scaled', ascending=False)

# 3. Görselleştirme Şovu (Yan Yana Katsayılar)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# SOL GRAFİK: Ham Veri Katsayıları
sns.barplot(x='Ham Veri Katsayısı (Unscaled)', y='Özellik', data=compare_coefs, ax=axes[0], palette='Reds_r')
axes[0].set_title(f"Standartlaştırma YOK (Ham Veri)\nTest R2: % {unscaled_r2*100:.2f}", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Katsayı (Birimlere Bağımlı)")
axes[0].axvline(0, color='black', linewidth=1)
axes[0].grid(True, linestyle='--', alpha=0.5)

# SAĞ GRAFİK: Ölçeklenmiş Veri Katsayıları
sns.barplot(x='Ölçeklenmiş Veri Katsayısı (Scaled)', y='Özellik', data=compare_coefs, ax=axes[1], palette='Blues_r')
axes[1].set_title(f"Standartlaştırma VAR (StandardScaler)\nTest R2: % {results_df.loc['Lasso Regression (CV)', 'R2_Score']*100:.2f}", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Katsayı (Adil Karşılaştırma - Standart Sapma Etkisi)")
axes[1].axvline(0, color='black', linewidth=1)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("\nKatsayı Karşılaştırma Tablosu:")
print(compare_coefs.drop('Mutlak_Scaled', axis=1).round(5))
#####################################################################################################3
#####################################################################################################
# AYKIRI DEĞERLERİN OLS ÜZERİNDEKİ KALDIRAÇ ETKİSİ (LEVERAGE EFFECT) GÖRSELLEŞTİRMESİ
#####################################################################################################

print("\n--- OLS Doğrusu Üzerindeki Kaldıraç Etkisi (Leverage Effect) Analizi ---")

feature = 'Caffeine consumption'
target = 'Sleep efficiency'


X_kirli = df_OUTLİER[[feature]]
y_kirli = df_OUTLİER[target]

ols_kirli = LinearRegression()
ols_kirli.fit(X_kirli, y_kirli)
kirli_skor = ols_kirli.score(X_kirli, y_kirli)

X_temiz = df_final[[feature]]
y_temiz = df_final[target]

ols_temiz = LinearRegression()
ols_temiz.fit(X_temiz, y_temiz)
temiz_skor = ols_temiz.score(X_temiz, y_temiz)

# === GÖRSELLEŞTİRME ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sol Grafik: Kirli Verideki OLS Çizgisi
sns.regplot(x=X_kirli[feature], y=y_kirli, ax=axes[0],
            scatter_kws={'alpha':0.5, 'color':'red'},
            line_kws={'color':'black', 'linewidth':3})
axes[0].set_title(f"Aykırı Değerler VAR (Orijinal Veri)\nEğitim Skoru (R2): % {kirli_skor*100:.1f}\nDoğru sağa doğru çekiliyor", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Kafein Tüketimi (mg)")
axes[0].set_ylabel("Uyku Verimliliği")
axes[0].grid(True, linestyle='--', alpha=0.5)

# Sağ Grafik: Temiz Verideki OLS Çizgisi
sns.regplot(x=X_temiz[feature], y=y_temiz, ax=axes[1],
            scatter_kws={'alpha':0.5, 'color':'green'},
            line_kws={'color':'black', 'linewidth':3})
axes[1].set_title(f"Aykırı Değerler BASKILANDI (df_final)\nEğitim Skoru (R2): % {temiz_skor*100:.1f}\nDoğru ana kütleye (merkeze) oturdu", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Kafein Tüketimi (mg)")
axes[1].set_ylabel("Uyku Verimliliği")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



################################################################################################################################################################################333


print("\n--- LASSO MODELİ GEÇERLİLİK (DIAGNOSTIC) ANALİZİ ---")

# Hataları (Residuals) hesaplıyoruz
# lasso_pred: Senin daha önce oluşturduğun Lasso test tahminleri
# y_test: Test setindeki gerçek değerler
residuals = y_test - lasso_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Gerçek vs Tahmin (Actual vs Predicted)
sns.scatterplot(x=y_test, y=lasso_pred, ax=axes[0], color='royalblue', alpha=0.7, s=60)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # y=x mükemmel tahmin çizgisi
axes[0].set_title("1. Gerçek vs Tahmin Edilen Değerler", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Gerçek Uyku Verimliliği (y_test)")
axes[0].set_ylabel("Tahmin Edilen (y_pred)")
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. Hata Dağılımı (Homoscedasticity Analizi)
sns.scatterplot(x=lasso_pred, y=residuals, ax=axes[1], color='darkorange', alpha=0.7, s=60)
axes[1].axhline(0, color='red', linestyle='--', lw=2) # 0 Hata çizgisi
axes[1].set_title("2. Hata (Residual) Dağılımı\n(Varyans Homojenliği)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Tahmin Edilen Değerler")
axes[1].set_ylabel("Hata Miktarı (Gerçek - Tahmin)")
axes[1].grid(True, linestyle='--', alpha=0.5)

# 3. Hataların Normal Dağılımı (Normality of Residuals)
sns.histplot(residuals, kde=True, ax=axes[2], color='seagreen', bins=20)
axes[2].set_title("3. Hataların Normal Dağılımı", fontsize=13, fontweight='bold')
axes[2].set_xlabel("Hata Miktarı")
axes[2].set_ylabel("Frekans")
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
######################################################################333

print("\n--- RANSAC Algoritması: Inlier ve Outlier Tespit Şovu ---")

temsilci_ozellik = 'Sleep duration'
feature_idx = X.columns.get_loc(temsilci_ozellik)

x_plot = X_train_scaled.iloc[:, feature_idx]

inlier_mask = ransac_model.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

plt.figure(figsize=(10, 6))

plt.scatter(x_plot[inlier_mask], y_train[inlier_mask],
            color='mediumseagreen', marker='o', s=70, alpha=0.8, edgecolors='white',
            label='Inliers (Geçerli Ana Kütle)')

plt.scatter(x_plot[outlier_mask], y_train[outlier_mask],
            color='crimson', marker='X', s=90,
            label='Outliers (RANSAC Tarafından Dışlananlar)')

plt.title(f"RANSAC'ın Aykırı Değer (Outlier) Yönetimi\n(Temsilci Özellik: {temsilci_ozellik})", fontsize=15, fontweight='bold')
plt.xlabel(f"Standartlaştırılmış {temsilci_ozellik}", fontsize=12)
plt.ylabel("Uyku Verimliliği (Hedef Değişken)", fontsize=12)
plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
