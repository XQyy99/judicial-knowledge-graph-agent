[README.md](https://github.com/user-attachments/files/27986886/README.md)

`new_train.json`和`new_test.json`分别是用于训练和测试的数据集。文件夹下的几个`.ipynb`为进行数据处理等操作时使用的代码文件。在`NER`和`RE`两个文件夹里面，分别存储了用于命名实体识别和关系提取的训练以及测试代码。

在`databases`文件夹中，存储了最终的`Neo4j`数据库。为了执行最终的案例检索系统，首先需要确保在本地环境安装了`neo4j == 3.5.5`，然后执行以下命令导入到本地的`Neo4j`数据库当中：
```
neo4j-admin load --from=databases\graph.db.dump --database=graph.db --force
```
接下来，切换到`app`文件夹当中，然后执行：
```
python app.py
```
**注意**：运行之前需要修改`app.py`的`neo4j`账号和密码为运行环境的账号密码，并且需要修改`app.y`当中的`API_KEY`为自己的openai key，以便于ChatGPT的调用。

在`邓智杰毕业设计文档.pdf`当中也详细描述了关键代码的逻辑。
