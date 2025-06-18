
#!/usr/bin/env python3
"""
Classificador de Grãos - Script Standalone
Desenvolvido usando metodologia CRISP-DM
Acurácia: 97.8%

Uso:
    python classificador_graos.py
"""

import pickle
import numpy as np
from datetime import datetime

class ClassificadorGraos:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.encoder = None
        self.info = None
        self.carregar_modelo()
    
    def carregar_modelo(self):
        """Carrega o modelo e componentes"""
        try:
            self.modelo = pickle.load(open('modelo_treinado.pkl', 'rb'))
            self.scaler = pickle.load(open('scaler.pkl', 'rb'))
            self.encoder = pickle.load(open('label_encoder.pkl', 'rb'))
            self.info = pickle.load(open('modelo_info.pkl', 'rb'))
            print(f"✅ Modelo carregado - Acurácia: {self.info['acuracia']:.1%}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
    
    def classificar(self, area, perimeter, compactness, length_kernel, 
                   width_kernel, asymmetry, groove_length):
        """Classifica um grão"""
        if self.modelo is None:
            return None, None, None
        
        features = [[area, perimeter, compactness, length_kernel, 
                    width_kernel, asymmetry, groove_length]]
        features_scaled = self.scaler.transform(features)
        
        predicao = self.modelo.predict(features_scaled)[0]
        probabilidades = self.modelo.predict_proba(features_scaled)[0]
        
        variedade = self.encoder.classes_[predicao]
        confianca = max(probabilidades)
        
        return variedade, confianca, probabilidades

def main():
    """Função principal para uso interativo"""
    classificador = ClassificadorGraos()
    
    print("🌾 CLASSIFICADOR DE GRÃOS DE TRIGO")
    print("="*40)
    print("Características necessárias:")
    print("1. Área")
    print("2. Perímetro") 
    print("3. Compacidade")
    print("4. Comprimento do núcleo")
    print("5. Largura do núcleo")
    print("6. Assimetria")
    print("7. Comprimento do sulco")
    print()
    
    try:
        area = float(input("Área: "))
        perimeter = float(input("Perímetro: "))
        compactness = float(input("Compacidade: "))
        length_kernel = float(input("Comprimento do núcleo: "))
        width_kernel = float(input("Largura do núcleo: "))
        asymmetry = float(input("Assimetria: "))
        groove_length = float(input("Comprimento do sulco: "))
        
        variedade, confianca, probs = classificador.classificar(
            area, perimeter, compactness, length_kernel,
            width_kernel, asymmetry, groove_length
        )
        
        if variedade:
            print(f"\n🎯 RESULTADO:")
            print(f"   Variedade: {variedade}")
            print(f"   Confiança: {confianca:.1%}")
            print(f"\n📊 Probabilidades:")
            for i, classe in enumerate(classificador.encoder.classes_):
                print(f"   {classe}: {probs[i]:.1%}")
        else:
            print("❌ Erro na classificação")
            
    except ValueError:
        print("❌ Erro: Digite apenas números")
    except KeyboardInterrupt:
        print("\n👋 Programa encerrado")

if __name__ == "__main__":
    main()
