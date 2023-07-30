# ImageProcessingEngine

このリポジトリでは画像処理エンジンの開発を行います。
超解像や画像自動生成も行うために、自作AIもこのエンジンに組み込んでいます。
正確には「画像処理＋AI＋シェーダ」エンジンです。

## ソリューション構成
### 画像処理エンジン
画像処理を行うためのエンジンです。基本的にはOpenCVを使って画像の処理を行います。

### AI
元々 https://github.com/asahi-kojima/DeepLearningCpp に置いていたプロジェクトです。  
アーキテクチャの改良に伴い、こちらに移行しました。

### グラフィックス＆コンピュートシェーダ
GPU処理と描画が出来るようにDirectx12を用いたレンダーエンジンを作っているプロジェクトです。
