# Document Cleanup

## Rodar o projeto

Para rodar o projeto você deve utilizar o seguinte comando:

```
python main.py 01-DocumentCleanup/noisy_data/
```

Lembre-se de instalar as bibliotecas necessárias contidas em requirements.txt!


## Resultados Alcançados

Os resultados alcançados podem ser conferidos na pasta "results".

## Técnicas utilizadas

Para realizar o desafio, foram utilizados as seguintes técnicas:

- <b>Filtragem</b>: O filtro de Mediana foi utilizado para remover os ruídos das imagens;

- <b>Operações Morfológicas</b>: O filtro morfológico foi utilizado para refinar a eliminação de ruídos e destacar os dígitos;

- <b>SIFT e minAreaRect</b>: Para calcular os ângulos dos textos, foi utilizado o algoritmo SIFT para encontrar pontos-chave na imagem e um minAreaRect foi estimado a partir destes pontos para obter o valor so ângulo dos textos e, desta forma, poder rotacioná-los.