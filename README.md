# WonderEngine

このリポジトリでは画像処理エンジンの開発を行います。
超解像や画像自動生成も行うために、自作AIもこのエンジンに組み込んでいます。
正確には「画像処理＋AI＋シェーダ＋統計解析」エンジンです。

## ソリューション構成
### ImageProcessingEngine
画像処理を行うためのエンジンです。

### AIEngine
元々 https://github.com/asahi-kojima/DeepLearningCpp に置いていたプロジェクトです。  
アーキテクチャの改良に伴い、こちらに移行しました。  
自動微分などが出来るように設計しています。

### RenderEngine
GPU処理と描画が出来るようにDirectx12を用いたレンダーエンジンを作っているプロジェクトです。
CUDA系もここに入れると思います。

### StatisticsEngine
統計解析を行うためのエンジンです。

