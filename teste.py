import joblib

caminho = r"D:\hd_central\projetos\thermal_vision_workspace\models\hybrid_stacking_model.joblib"

modelo = joblib.load(caminho)

print(type(modelo))
print(modelo)

import joblib

classes = joblib.load(
    r"D:\hd_central\projetos\thermal_vision_workspace\models\class_names.joblib"
)

print(classes)