"""Stage 2：防御性扰动生成（成员 B 主责，任务板 W4–W5）。

【这一阶段在做什么】
论文第 2.3 节。Stage 1 已经把语音拆成了「内容码 c」和「说话人嵌入 h」，
Stage 2 只干一件事：

    学一个生成器 G，让它在说话人嵌入空间里挑一个**人耳听不出来、
    但克隆系统能听出来**的方向，然后沿这个方向推一下：

        δ    = G(h)                  ← 可训练
        h_adv = h + α·δ              ← α = 0.10
        x_adv = D(c, h_adv)          ← Stage 1 的解码器（冻结）

    攻击者拿保护后的语音重新提卡（h_stolen）再去克隆，
    克隆出来的声音就不像本人了 —— 这就是防御。

【模块分工】
    common.py          扰动、余弦相似度、随机种子等公共件
    generator.py       生成器 G（h → δ）
    discriminator.py   判别器 D_adv（mel → 真假分数），LSGAN
    psychoacoustics.py 心理声学掩蔽阈值 θ_x（论文式 8）
    losses.py          L_mel / L_mask / L_per / L_GAN / L_D / L_de / L_G
    models.py          **可求导**的冻结组件（GE2E、Stage 1 解码器、声码器）
    dataset.py         VCTK 数据集
    train.py           训练主循环

【★ 一个必须知道的坑：冻结 ≠ 断梯度】
Stage 1 的解码器 D、声码器 V、说话人编码器 E_spk 都**冻结**
（不更新它们的参数），但**梯度必须能穿过它们流回 G**：

    G → h_adv → D(c, h_adv) → V(...) → E_spk(...) → L_de

所以 Stage 2 **不能**用 `bridge_autovc.protect()` / `bridge_vocoder.synthesize()`
（那两个是 numpy 封装，内部 `torch.no_grad()`，梯度被切断了）。
`models.py` 里提供的是 **torch 原生、可求导**的等价实现。
"""
