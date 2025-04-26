import os
import sys
import subprocess
import platform

def main():
    """Script para configurar o ambiente de desenvolvimento automaticamente."""
    
    print("🚀 Configurando ambiente para o projeto de Regressão Linear...")
    
    # Verifica se o Conda está instalado
    try:
        subprocess.run(['conda', '--version'], check=True)
    except:
        print("❌ Conda não encontrado. Por favor, instale o Anaconda ou Miniconda primeiro.")
        sys.exit(1)
        
    # Cria o ambiente conda usando o arquivo environment.yml
    print("\n📦 Criando ambiente conda...")
    subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], check=True)
    
    # Instruções de ativação baseadas no sistema operacional
    if platform.system() == "Windows":
        activate_cmd = "conda activate regressao-linear-ex1"
    else:
        activate_cmd = "source activate regressao-linear-ex1"
        
    print("\n✅ Ambiente configurado com sucesso!")
    print("\nPara ativar o ambiente, execute:")
    print(f"\n    {activate_cmd}")
    print("\nDepois execute:")
    print("    python regressao-linear-ex1.py")

if __name__ == "__main__":
    main()