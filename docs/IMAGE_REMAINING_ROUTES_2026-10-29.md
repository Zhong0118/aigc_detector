# 剩余图片路线登记与资产审计

本文件对应用户在页面交接后继续实验的授权。既有100/200版、拓展、冻结OmniAID/OpenCLIP和比较均已完成；不要重跑。新增结果沿用200-q179与旧角色，holdout不推理。网页实施交给其他模型，见IMAGE_UI_HANDOFF_2026-10-29.md。

## 本轮直接实施
ConvNeXt-Tiny官方ImageNet22k→1k权重：timm/convnext_tiny.fb_in22k_ft_in1k固定bc48a87f119bae1b06418f5340bfa97ef75d4609，114374272字节，SHA77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a，模型卡Apache2。官方测试288/bicubic/crop1.0/ImageNet归一化；完整1000类strict load后reset_classifier(0)，768维L2。依次D-clean、D-heldFFHQ、A22，冻结骨干，新线性头各30轮。不是专门伪造检测论文的完整复现。

入口scripts/run_image_convnext.py；配置iab_convnext_q179_20261029.json；输出convnext-q179-20261029。主要同协议参考OpenCLIP；ResNet检测是训练方式/预处理等均不同的整路线参考。A0来源模型曾用包含evaluation的整个parent-dev选模，故其比较单独放iab_convnext_parent_dev_reference_20261029.json，不列为相同选模协议主排名。复用原compare_image_next_routes.py做完整ID/角色/标签/词表校验与1000次group bootstrap，不按交集丢样本。

## OmniDFA：可继续冻结资产适配，未等同已运行
[官方代码](https://github.com/teheperinko541/OmniDFA/tree/35b9052e83e05436682095818693493f79da9458)，[权重](https://huggingface.co/MoeNew/OmniDFA/tree/b4c0f420461f154c0d71f1abc13fa01e11e4975b)。事前选part1 epoch20，不按本地eval挑fold；1198070905字节，SHA91d368bbd24a20a3cbe9b1f381e334580cef7f4c1c10f7d058961371945d1e33。HF卡Apache2，代码未见LICENSE，不能另称MIT或直接当可商用部署依据。

模型双独立ConvNeXt-Small，concat→MLP512→128维，forward(local,global)。预处理需最小短边256保证：低于256时随机选PIL插值放大，local RandomCrop224，global tensor bilinear短边Resize224再RandomCrop224，无mean/std。缓存复现按sample_id固定随机种子，记录与作者随机episode的区别；不能遗漏短边保护，当前非holdout最小短边90。

冻结预训练特征＋A22线性头30轮和固定每类10个train参考的归一化均值prototype可作项目适配，prototype epoch=null，不冒称原5/15way成绩或未知拒识。作者trainlist和q179有来源重叠，具体版本/图像重叠未建立。checkpoint为dill容器，下载后先size/SHA与静态globals检查，始终安全加载，不直接照抄作者weights_only=False。

## DNA-Det：冻结作者骨干适配与完整方法分开
[代码固定commit](https://github.com/ICTMCG/DNA-Det/tree/d811395b92fb22d5f82f06ee98dbde77e53cea69)。[in-the-wild权重](https://drive.google.com/file/d/1bvFXsNgEs43JH50L6eLAqbdzZCBKO9j6/view)公开HEAD为23635063字节、CRC32C MrAURw==；[变换预训练权重](https://drive.google.com/file/d/11MQxUXW1NAlBvOvxpoSaiqRh2nGmRHjy/view)23959671字节、CRC32C dj7ySw==。无发布者SHA/不可变revision，导入必须保存fileID/时间/响应size及实际SHA；来源与权重许可未建立，保留此局限。

最轻路线是原11类Simple_CNN完整strict load后提取pool(embedding).flatten的512维特征，冻结并接22来源30轮新头。输入RGB，非正方形先CenterCrop128，resize128再512，mean/std .5，完整图评估。必须称“DNA-Det预训练骨干线性适配”，不是完整CE+SupCon+AWL训练。官方balance模式batch32表示每类32组，22类×32×16patch=11264patch/step，不是32图片；完整训练另列资源协议。官方将softmax后结果交CrossEntropyLoss，若修正为logits也须披露。原任务GAN架构标签与本项目发布目录标签不等价。

## OFA全预训练与T+1：有新入口但仍需资产/预算门槛
现有冻结OFA已完成，不能重跑或改原结果。[DAIR固定代码](https://github.com/jumpycat/DAIR/tree/1ec4fd41d70270fc20f62cbd716491b29fc7e6ee)README确有[模拟器权重入口](https://drive.google.com/file/d/1v5V1u6873oNj1sSwURqnhIK4A3r3QrM1/view)，期待250708204953-4m.pth的encoder/decoder键。当前只核验链接存在，尚未核验真实文件/大小/SHA/加载，旧“未找到公开入口”应更新为此状态。

DAIR默认COCO真图、crop256、MSE+.25LPIPS(VGG)、batch4累计32、100万microsteps；OFA提取器默认10万steps、batch16且每项8架构，多种输入分支；论文memorybank1024与代码65536不同。不能归入30epoch分类头预算，也不能凭batch16估显存。固定OFA代码train_noiser_fft_bank.py对decoder(latents).detach()调用，而decoder24_with_graph.py返回(output,arch)，存在静态接口不一致；缺模拟器还可能继续使用未初始化encoder，启动前必须复现修正/失败保护。

[T+1论文机制](https://ojs.aaai.org/index.php/AAAI/article/download/40851/44812)以模拟器重建真实图训练unknown类；官方tree未含专用T+1入口。可在模拟器验收后固定train-only真实图/架构采样，接冻结OFA特征做20+1头30轮，沿原20known/kling-cal/grok3-eval比较。明确是本项目适配，不把原真实图、kling或者一个拒识阈值替代模拟未知类，不宣称全预训练已实现。

## A3：部分证据建立，完整家族映射继续blocked
旧iab_family_mapping_review.json保持reviewed=false，不改旧模板和结果。
- SD1_5/SD2_1/SD3/SD3_5/SDXL可归Stable Diffusion产品系列；[作者SD生成代码](https://github.com/mttry/ImageAttributionBench/blob/19bb7d805142d30e67689dae009ece170866c5f3/dataset_construction/t2i_generator/diffuser_models/SDModel.py#L60)有精确仓库。系列不等于相同架构，SD3/3.5含DiT变化。
- mid-5.2/mid-6.0可归Midjourney产品系列；[作者版本参数与目录](https://github.com/mttry/ImageAttributionBench/blob/19bb7d805142d30e67689dae009ece170866c5f3/dataset_construction/t2i_generator/API/mid_gen_image.py#L350)支持版本身份，不证明后端同架构。
- 4o/dalle3/gemini/grok3/ideogram/kling不可因供应商/名字强行归并。其余单版本来源没有因此获得跨版本家族泛化证据。

作者dataset_hifi_net.py的commercial/SD/diffusers/DiT/AR混合商业属性、库与架构，不是统一家族定义，不能照抄为A3已核验。后续若展示部分产品系列，只能另建明确“未核验”覆盖统计的协议，不能伪称全部22类架构家族。

## 验收和归档
源代码、配置、资产、数据和角色SHA先登记；任何资产下载/安全加载失败保留文件和诊断，不绕TLS或静默unsafe fallback。新输出独占创建，failed不重置。完成检查真实history1..30（冻结/prototype例外null）、重载/单批一致≤1e-5、predictions角色和ID完整、无holdout；保存源路径/哈希、metrics/history/逐类/配对结果到独立Git快照，不上传模型/图片/特征/逐图大预测。

## 本轮后续核验补充
DNA-Det官方权重现已完整导入并CPU安全strict-load，实际SHA641cf077f667636c62828227cd1d4ecb78494744245186062973b0f31d962d92；独立新入口run_image_dna.py已通过47项CPU测试/独立审查，已绑定CPU收据，尚未做正式q179GPU推理。OmniDFA part1的1198070905字节和发布者SHA已全文件核验，静态extra globals仅argparse.Namespace，安全适配开发中。
用户已明确优先官方模拟器＋20+1拒识头30轮。OFA模拟器138627153字节SHA7ee94487217a8ff9fad3b0da1bdbb60b89f923be09b377b5f1ef17d9e8eb3de8通过weights_only及官方encoder/decoder CPU严格加载；未运行完整预训练。另发现官方arch_json回放会把浅层架构误解为深层（CPU合成输入复现），本轮使用隔离Python RNG种子回放并核验真实trace，保留上游源码不修改。重建未知类预登记1024个不同real-train组、8架构各128张；20+1头只以known selection选epoch，kling仅cal，grok3仅eval。这些资产/合成CPU检查均不是模型成绩。
