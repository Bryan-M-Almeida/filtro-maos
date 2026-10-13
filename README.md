# Filtro de mãos

Protótipo local em Python: a webcam mantém o vídeo normal e aplica efeitos apenas dentro da moldura formada por duas mãos em L. A galeria alterna a cada 6 segundos entre Pop Art/HQ americana, filme clássico americano, Art Nouveau europeu e retrato real dourado. Para expandir o filtro, feche as duas mãos e depois abra-as rapidamente; para recolher, feche as mãos devagar. O efeito volta para as palmas e depois pode voltar à moldura dos dois Ls. Ao formar ou mover a moldura, aparecem por um instante onomatopeias como POW!, BAM! e THWIP!. O nome do estilo aparece numa plaquinha acima do rosto. Uma palma avança para o próximo estilo. Também dá para navegar com a mão aberta: faça um tapa rápido para a direita para avançar e para a esquerda para voltar. O estilo escolhido fica pausado por 12 segundos antes da alternância automática continuar. O áudio só é analisado localmente para detectar picos; não é gravado nem enviado.

## Como iniciar no Windows

1. Instale Python 3.10 ou 3.11, se ainda não tiver.
2. Abra o PowerShell nesta pasta.
3. Instale as dependências:

   ```powershell
   py -m pip install -r requirements.txt
   ```

4. Inicie:

   ```powershell
   py filtro_maos.py
   ```

Na primeira execução, o programa baixa o modelo oficial de detecção de mãos do MediaPipe para esta pasta. A câmera e o microfone precisam estar disponíveis; se o microfone não abrir, o filtro de câmera continua funcionando. Para iniciar sem microfone, use `py filtro_maos.py --no-mic`. O programa abre em tela cheia; pressione M para alternar entre tela cheia e janela. A galeria inicia ligada; faça dois Ls com os polegares e indicadores para enquadrar a área dos efeitos. Os estilos mudam automaticamente a cada 6 segundos; um tapa lateral com a mão aberta avança ou volta entre eles. Dois punhos fechados seguidos de duas mãos abertas fazem o filtro explodir pela tela toda; feche as mãos aos poucos para encolher o efeito de volta às palmas. Pressione F para ligar/desligar a galeria e Q ou Esc para fechar.

O programa tenta abrir o índice selecionado com DirectShow, Media Foundation e o modo automático; se necessário, tenta outros índices. Para priorizar o DroidCam, tente `py filtro_maos.py --camera 1`, depois `--camera 2` ou `--camera 3`. Deixe o cliente DroidCam conectado e feche apps que possam estar usando a câmera. Para ajustar a sensibilidade do microfone, use `--threshold 0.24` (mais sensível) ou `--threshold 0.45` (menos sensível).
