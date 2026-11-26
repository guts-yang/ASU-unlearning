# -*- coding: utf-8 -*-
"""LaTeX formula specs for the 10 CF idea mechanism diagrams.

All formula strings are Python raw strings so LaTeX backslashes survive.
`$$...$$` is the drawio (math="1") display-math delimiter, exactly as in the
reference file ASU_遗忘全链路实现图_代码实证版v2.drawio.
"""

def C():
    return {"in": "#EEEDFE", "in_s": "#534AB7", "inR": "#E6F1FB",
            "inR_s": "#185FA5", "donor": "#FAEEDA", "donor_s": "#BA7517"}

# common skeleton fields reused by every idea
def base(iid, folder, title, **kw):
    d = dict(
        id=iid, folder=folder, title=title, colors=C(),
        forget="遗忘集合  D_F", forget_f=r"$$\mathcal{D}_F=\{(X^{+},y^{+})\}$$",
        retain="保留集合  D_R", retain_f=r"$$\mathcal{D}_R=\{(X_r,y_r)\}$$",
        donor="反事实配对  X⁻", donor_f=r"$$X^{-}=X(s_r,r)$$",
        embed=r"$$h^{(0)}=E[\mathrm{ids}]$$",
        t_plus=r"$$A^{+}_{T,\ell}(p)=h_{T,\ell}(X^{+}\!\oplus y^{+};p)$$",
        t_minus=r"$$A^{-}_{T,\ell}(p)=h_{T,\ell}(X^{-}\!\oplus y^{+};p)$$",
        delta=r"$$\Delta A_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,\ell}(p),\;p\in S_f$$",
        s_fwd=r"$$h^{(\ell)}=\mathrm{Dec}_{\ell}(h^{(\ell-1)};\theta)$$",
        s_act=r"$$h_{S,\ell}(X^{+}\!\oplus y^{+};p),\;p\in S_f$$",
        target=r"$$T_{\ell,p}=\operatorname{stopgrad}(\cdot)$$",
        lossF=r"$$\mathcal{L}_F=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\,\operatorname{dist}\!\left(h_{S,\ell}(p),T_{\ell,p}\right)$$",
        lossR=r"$$\mathcal{L}_R=\mathbb{E}_{(x,y)\sim\mathcal{D}_R}\!\left[-\log p_\theta(y\mid x)\right]$$",
        total=r"$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \lambda_{KL}^{(\mathrm{ablation})}=0$$",
        backprop=r"$$\nabla_{\theta}\mathcal{L}\;\text{only},\quad \theta_0\ \text{frozen (stopgrad)},\quad \theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}$$",
        output=r"$$\theta^{*}\ \text{：参数内化、无 hook、具备遗忘且保效用的 Student 模型}$$",
        detail=[],
    )
    d.update(kw)
    return d


SPECS = [
    # ---------------- CF-01 约束反演目标 ----------------
    base("CF-01", "CF-01_约束反演目标",
         "CF-01 约束反演目标 Δ* —— 因果+效用+身份三约束逆问题（机制流程）",
         delta=r"$$\Delta A_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,\ell}(p),\;p\in S_f\quad(\text{仅初始化/负约束})$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(h^{+}_{\ell}(p)+\Delta^{*}_{\text{causal},\ell,p}\right)$$",
         detail=[
             {"h": "① 双前向取激活差", "f": r"$$A^{\pm}_{T,\ell}(p)=h_{T,\ell}(X^{\pm}\!\oplus y^{+};p),\quad \Delta A_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,\ell}(p),\;p\in S_f$$", "w": 300},
             {"h": "② 约束反演 求解 Δ*", "f": r"$$\Delta^{*}_{\ell,p}=\arg\min_{\Delta}\; r_f\!\left(h^{+}_{\ell}(p)+\Delta\right)+\gamma\lVert\Delta\rVert^{2}$$$$\text{s.t.}\; r_f\!\left(h^{+}+\Delta\right)\le c,\; U\!\left(h^{+}+\Delta\right)\ge U\!\left(h^{+}\right)-\delta,\; \langle\Delta,v_{\mathrm{id}}\rangle\le 0,\; \Delta\in\mathrm{Null}(G^{\top}_{\mathrm{retain}}),\; \lVert\Delta\rVert\le\varepsilon$$", "w": 430},
             {"h": "③ 中性约束（非 donor）", "f": r"$$r_{f,p}(h)\le c,\quad s_{d_j,p}(h)\le d_j\;(\forall j),\quad \lVert\Delta r_{E,p}(h)\rVert\le e,\quad p\in S_f$$", "w": 300},
             {"h": "④ patch 因果验证", "f": r"$$\Delta^{*}_{\mathrm{causal}}=\sum_{\ell,p} w_{\ell,p}\Delta^{*}_{\ell,p},\quad w_{\ell,p}=\mathbb{1}\!\left[\mathrm{Eff}^{\mathrm{patch}}_{r_f}(\ell,p)<-\eta_c\ \wedge\ |\mathrm{Eff}^{\mathrm{patch}}_{U}(\ell,p)|<\eta_U\right]$$", "w": 330},
             {"h": "⑤ 目标与前向", "f": r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(h^{+}_{\ell}(p)+\Delta^{*}_{\mathrm{causal},\ell,p}\right),\quad h_{S,\ell}(X^{+}\!\oplus y^{+};p)$$", "w": 300},
             {"h": "⑥ 损失 / 反传 / 输出", "f": r"$$\mathcal{L}_F(k)=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\,\operatorname{dist}\!\left(h_{S,\ell}(p),T_{\ell,p}\right)$$$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \nabla_{\theta}\mathcal{L}\ \text{仅更新}\ \theta,\quad \theta_0\ \text{frozen},\quad \Rightarrow\ \theta^{*}$$", "w": 360},
         ]),

    # ---------------- CF-02 三前向角色分解 ----------------
    base("CF-02", "CF-02_三前向角色分解",
         "CF-02 三前向角色分解 —— 删什么 / 别变成什么 / 保什么（机制流程）",
         t_minus=r"$$A^{-}_{T,\ell}(p)=h_{T,\ell}(X^{-}\!\oplus y^{+};p),\quad h^{r}_{T,\ell}(p)=h_{T,\ell}(X_r\!\oplus y_r;p)$$",
         delta=r"$$\Delta A=\Pi_{\text{fact}}\Delta A+\Pi_{\text{id}}\Delta A+\Pi_{\text{util}}\Delta A+\epsilon$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(h^{+}_{\ell}(p)+\Delta^{*}_{\ell,p}\right)$$",
         detail=[
             {"h": "① 三路前向（增 retain）", "f": r"$$h^{+}_{\ell}(p)=h_{T,\ell}(X^{+}\!\oplus y^{+};p),\quad h^{-}_{\ell}(p)=h_{T,\ell}(X^{-}\!\oplus y^{+};p),\quad h^{r}_{\ell}(p)=h_{T,\ell}(X_r\!\oplus y_r;p),\quad p\in S_f$$", "w": 330},
             {"h": "② 三角色投影分解", "f": r"$$\Delta A_{\ell,p}=\underbrace{\Pi_{\text{fact}}\Delta A_{\ell,p}}_{\text{删什么}}+\underbrace{\Pi_{\text{id}}\Delta A_{\ell,p}}_{\text{禁止方向}}+\underbrace{\Pi_{\text{util}}\Delta A_{\ell,p}}_{\text{效用地板}}+\epsilon_{\ell,p}$$", "w": 300},
             {"h": "③ 三方约束反演", "f": r"$$\Delta^{*}_{\ell,p}=\arg\min_{\Delta} r_f\!\left(h^{+}_{\ell}(p)+\Delta\right)$$$$\text{s.t.}\; r_f(h^{+}+\Delta)\le c,\; \langle\Delta,\widehat{\Delta A}_{\text{donor}}\rangle\le-\kappa_d,\; U(h^{+}+\Delta)\ge U(h^{+})-\delta,\; s_{d_j,p}\le d_j,\; \lVert\Delta r_{E,p}\rVert\le e$$", "w": 430},
             {"h": "④ donor 仅作负约束", "f": r"$$\widehat{\Delta A}_{\text{donor}}=h^{-}_{\ell}(p)-h^{r}_{\ell}(p)\quad(\text{只进不等式，不进目标，不拼}y^{-})$$$$U\ \text{由第三次 retain 前向标定},\quad \lambda_{KL}^{(\mathrm{ablation})}=0$$", "w": 300},
             {"h": "⑤ 目标 / 损失 / 反传", "f": r"$$T_{\ell,p}=\operatorname{stopgrad}(h^{+}_{\ell}(p)+\Delta^{*}_{\ell,p}),\quad \mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \nabla_{\theta}\mathcal{L}\Rightarrow\theta^{*}$$", "w": 330},
         ]),

    # ---------------- CF-03 因果门控 ----------------
    base("CF-03", "CF-03_因果门控_activation_gate",
         "CF-03 因果门控 ΔA —— patch 加权因果分量（机制流程）",
         delta=r"$$\Delta A^{\text{causal}}_{\ell,p}=w_{\ell,p}\,\Delta A_{\ell,p}$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A^{\text{causal}}_{\ell,p}}\right]\right)$$",
         detail=[
             {"h": "① 原始配对差", "f": r"$$\Delta A_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,\ell}(p),\quad p\in S_f,\quad \theta_0\ \text{stopgrad}$$", "w": 260},
             {"h": "② 逐分量 patch 因果效应", "f": r"$$\mathrm{Eff}_{r_f}(\ell,p)=r_{f,p}\!\left(\mathrm{PatchFwd}(\theta_0;h\mapsto h+\Delta A_{\ell,p})\right)-r_{f,p}\!\left(\mathrm{Fwd}(\theta_0)\right)$$$$\mathrm{Eff}_{U}(\ell,p)=U\!\left(\mathrm{PatchFwd}(\theta_0;h\mapsto h+\Delta A_{\ell,p})\right)-U\!\left(\mathrm{Fwd}(\theta_0)\right)$$", "w": 400},
             {"h": "③ 因果门控权重", "f": r"$$w_{\ell,p}=\sigma\!\left(\frac{-\mathrm{Eff}_{r_f}(\ell,p)-\eta_f}{t_f}\right)\cdot\sigma\!\left(\frac{\eta_U-|\mathrm{Eff}_{U}(\ell,p)|}{t_U}\right),\quad \Delta A^{\text{causal}}_{\ell,p}=w_{\ell,p}\Delta A_{\ell,p}$$", "w": 330},
             {"h": "④ 中性可行集投影", "f": r"$$\mathcal K=\{h:\ r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \lVert\Delta r_{E,p}(h)\rVert\le e\}$$$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A^{\text{causal}}_{\ell,p}}\right]\right)$$", "w": 330},
             {"h": "⑤ 损失 / 反传 / 输出", "f": r"$$\mathcal{L}_F(k)=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\operatorname{dist}\!\left(h_{S,\ell}(p),T_{\ell,p}\right)$$$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \nabla_{\theta}\mathcal{L}\ \text{更新}\ \theta,\quad \Rightarrow\theta^{*}$$", "w": 340},
         ]),

    # ---------------- CF-04 双探针解耦 ----------------
    base("CF-04", "CF-04_事实身份双探针解耦",
         "CF-04 事实/身份双探针正交解耦 ΔA_fact（机制流程）",
         delta=r"$$\Delta A_{\text{fact}}=(I-V_{\text{id}}V_{\text{id}}^{\top})\Delta A+\sum_j\lambda_j u_j$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{pot},\ell,p}}\right]\right)$$",
         detail=[
             {"h": "① 双冻结探针", "f": r"$$z_{\text{fact}}=\phi_{\text{fact}}(\Delta A_{\ell,p}),\quad z_{\text{id}}=\phi_{\text{id}}(\Delta A_{\ell,p}),\quad \mathrm{AUC}_f,\ \mathrm{AUC}_{\text{id}}$$$$\text{经冻结下游重算，不直接过 LM head}$$", "w": 300},
             {"h": "② 正交收缩去身份", "f": r"$$\Delta A_{\text{fact}}=\big(I-V_{\text{id}}V_{\text{id}}^{\top}\big)\Delta A,\quad \Delta A_{\text{id}}=V_{\text{id}}V_{\text{id}}^{\top}\Delta A,\quad p\in S_f$$", "w": 300},
             {"h": "③ 反事实激活潜力", "f": r"$$\Delta A_{\text{pot}}=\Delta A_{\text{fact}}+\sum_{j}\lambda_j u_j,\quad u_j\perp V_{\text{id}},\ \left|\partial r_f/\partial u_j\right|>\tau_{\text{pot}}$$$$\mathcal P=\mathrm{AUC}_f(\Delta A_{\text{pot}})-\mathrm{AUC}_{\text{id}}(\Delta A_{\text{pot}})$$", "w": 330},
             {"h": "④ 纯度证书（wrong-donor）", "f": r"$$\frac{\lVert V_{\text{id}}V_{\text{id}}^{\top}\Delta A^{\text{wrong}}\rVert}{\lVert\Delta A^{\text{wrong}}\rVert}\ge\pi_{\text{id}}$$$$\mathcal K=\{h: r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\}$$", "w": 320},
             {"h": "⑤ 目标 / 损失 / 反传", "f": r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{pot},\ell,p}}]\right)$$$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \nabla_{\theta}\mathcal{L}\Rightarrow\theta^{*}$$", "w": 340},
         ]),

    # ---------------- CF-05 多 donor 边际化 ----------------
    base("CF-05", "CF-05_多donor边际化",
         "CF-05 多 donor 边际化共有子空间 ΔA_marg（机制流程）",
         donor="K 个关系匹配 donor", donor_f=r"$$X^{-}_{(k)}=X(s_{r_k},r),\ k=1..K$$",
         t_minus=r"$$A^{-}_{T,(k),\ell}(p)=h_{T,\ell}(X^{-}_{(k)}\!\oplus y^{+};p)$$",
         delta=r"$$\Delta A_{\text{marg},\ell,p}=U^{(r)}_{\ell,p}U^{(r)\top}_{\ell,p}\,\overline{\Delta A}_{\ell,p}$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{marg},\ell,p}}\right]\right)$$",
         detail=[
             {"h": "① K-donor 配对差", "f": r"$$\Delta A^{(k)}_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,(k),\ell}(p),\quad k=1..K,\quad p\in S_f$$$$\text{donor 全来自 TOFU 内部保留集、按作者/别名簇隔离}$$", "w": 300},
             {"h": "② 堆叠与边际均值", "f": r"$$\overline{\Delta A}_{\ell,p}=\frac{1}{K}\sum_{k=1}^{K}\Delta A^{(k)}_{\ell,p},\quad M_{\ell,p}=\big[\Delta A^{(1)}_{\ell,p}\ \cdots\ \Delta A^{(K)}_{\ell,p}\big]$$", "w": 300},
             {"h": "③ SVD 共有子空间", "f": r"$$M_{\ell,p}=U^{(r)}_{\ell,p}\Sigma^{(r)}_{\ell,p}V^{(r)\top}_{\ell,p}+E_{\ell,p},\quad \Delta A_{\text{marg},\ell,p}=U^{(r)}_{\ell,p}U^{(r)\top}_{\ell,p}\,\overline{\Delta A}_{\ell,p}$$$$\Delta A^{(k)}_{\text{priv}}=\Delta A^{(k)}_{\ell,p}-\Delta A_{\text{marg},\ell,p}\ (\text{投影掉})$$", "w": 380},
             {"h": "④ 目标（非 donor）", "f": r"$$\mathcal K=\{h: r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\},\quad T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{marg},\ell,p}}]\right)$$", "w": 330},
             {"h": "⑤ 损失 / 反传 / 输出", "f": r"$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \lambda_{KL}^{(\mathrm{ablation})}=0,\quad \nabla_{\theta}\mathcal{L}\ \text{仅}\ \theta,\quad \Rightarrow\theta^{*}$$", "w": 300},
         ]),

    # ---------------- CF-06 LRM 推理轨迹 ----------------
    base("CF-06", "CF-06_LRM推理轨迹_deltaA",
         "CF-06 LRM 推理轨迹跨度 ΔA_trace —— 逐 CoT step（机制流程）",
         forget_f=r"$$\mathcal{D}_F=\{(X^{+},\tau^{+})\},\ \tau^{+}=(c^{+}_1..c^{+}_T)$$",
         t_plus=r"$$h_{T,\ell}(X^{+}\!\oplus\tau^{+};p_t),\quad t\in S_{\text{trace}}$$",
         t_minus=r"$$h_{T,\ell}(X^{-}\!\oplus\tau^{+};p_t),\quad t\in S_{\text{trace}}$$",
         delta=r"$$\Delta A_{\text{trace},\ell,t}=h_{T,\ell}(X^{+}\!\oplus\tau^{+};p_t)-h_{T,\ell}(X^{-}\!\oplus\tau^{+};p_t)$$",
         s_act=r"$$h_{S,\ell}(X^{+}\!\oplus\tau^{+};p_t),\quad t\in S_{\text{trace}}$$",
         target=r"$$T_{\ell,t}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p_t)-\alpha\widehat{\Delta A_{\text{trace},\ell,t}}]\right)$$",
         lossF=r"$$\mathcal{L}_F^{\text{trace}}(k)=\frac{1}{|\Lambda(k)|}\sum_{\ell\in\Lambda(k)}\sum_{t\in S_{\text{trace}}}\rho_\ell(k)\rho(t)\operatorname{dist}\!\left(h_{S,\ell}(p_t),T_{\ell,t}\right)$$",
         total=r"$$\mathcal{L}=\lambda_F\mathcal{L}_F^{\text{trace}}+\beta\mathcal{L}_R+\gamma\mathcal{L}_{\text{reason}},\quad \lambda_{KL}^{(\mathrm{ablation})}=0$$",
         detail=[
             {"h": "① 反事实轨迹对", "f": r"$$\tau^{+}=(c^{+}_1..c^{+}_T),\ \tau^{-}=(c^{-}_1..c^{-}_T),\quad S_{\text{trace}}=\{t:\ \text{step }t\text{ 含事实绑定}\}$$$$\text{由事实 probe + tokenizer offset 冻结，不随 schedule 变}$$", "w": 320},
             {"h": "② 逐 step 轨迹激活差", "f": r"$$\Delta A_{\text{trace},\ell,t}=h_{T,\ell}(X^{+}\!\oplus\tau^{+};p_t)-h_{T,\ell}(X^{-}\!\oplus\tau^{+};p_t),\quad t\in S_{\text{trace}}$$", "w": 300},
             {"h": "③ 数据驱动 step 权重", "f": r"$$\mathrm{Eff}_{r_f}(t)=\text{patch-forward 在 step }t\text{ 的因果读出变化},\quad \rho(t)=\frac{\exp(\mathrm{Eff}_{r_f}(t)/t_0)}{\sum_{t'\in S_{\text{trace}}}\exp(\mathrm{Eff}_{r_f}(t')/t_0)}$$", "w": 330},
             {"h": "④ 目标（中性约束）", "f": r"$$\mathcal K=\{h: r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\},\quad T_{\ell,t}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p_t)-\alpha\widehat{\Delta A_{\text{trace},\ell,t}}]\right)$$", "w": 330},
             {"h": "⑤ 损失 / 反传 / 输出", "f": r"$$\mathcal{L}_F^{\text{trace}}(k)=\frac{1}{|\Lambda(k)|}\sum_{\ell\in\Lambda(k)}\sum_{t\in S_{\text{trace}}}\rho_\ell(k)\rho(t)\operatorname{dist}\!\left(h_{S,\ell}(p_t),T_{\ell,t}\right)$$$$\mathcal{L}=\lambda_F\mathcal{L}_F^{\text{trace}}+\beta\mathcal{L}_R+\gamma\mathcal{L}_{\text{reason}},\quad \nabla_{\theta}\mathcal{L}\Rightarrow\theta^{*}\ (\text{保推理效用})$$", "w": 380},
         ]),

    # ---------------- CF-07 读出空间 ----------------
    base("CF-07", "CF-07_读出空间_deltaA_readout",
         "CF-07 读出空间 ΔA_readout —— 旋转不变（机制流程）",
         delta=r"$$\Delta A_{\text{readout},\ell,p}=\left(\Delta A^{\text{scalar}}_{\ell,p},\ \widehat{J}_{\ell,p}\right)$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\,\Delta A^{\text{scalar}}_{\ell,p}\,\widehat{J}_{\ell,p}\right]\right)$$",
         detail=[
             {"h": "① 冻结下游读出泛函", "f": r"$$r_f(h)=\mathcal R_{y^{+}}(h),\quad \Delta A^{\text{scalar}}_{\ell,p}=r_f\!\left(A^{+}_{T,\ell}(p)\right)-r_f\!\left(A^{-}_{T,\ell}(p)\right)$$$$\text{经 patch-forward 重算，不直接过 LM head}$$", "w": 330},
             {"h": "② Jacobian 局部方向", "f": r"$$J_{\ell,p}=\frac{\partial r_f(h)}{\partial h}\Big|_{h=A^{+}_{T,\ell}(p)},\quad \widehat{J}_{\ell,p}=J_{\ell,p}/\lVert J_{\ell,p}\rVert$$", "w": 270},
             {"h": "③ 读出空间 ΔA", "f": r"$$\Delta A_{\text{readout},\ell,p}=\big(\Delta A^{\text{scalar}}_{\ell,p},\ \widehat{J}_{\ell,p}\big)$$$$r_f(Qh)=r_f(h)\Rightarrow \Delta A^{\text{scalar}}_{\ell,p}(Q)=\Delta A^{\text{scalar}}_{\ell,p},\ Q^{\top}Q=I$$", "w": 300},
             {"h": "④ 旋转不变性 / 中性集", "f": r"$$Q A^{+}-Q A^{-}\ \ne\ A^{+}-A^{-}\ (\text{一般}),\quad \mathcal K=\{h: r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\}$$$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p)-\alpha\Delta A^{\text{scalar}}_{\ell,p}\widehat{J}_{\ell,p}]\right)$$", "w": 360},
             {"h": "⑤ 损失 / 反传 / 输出", "f": r"$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \nabla_{\theta}\mathcal{L}\ \text{仅}\ \theta,\quad \theta_0\ \text{frozen},\quad \Rightarrow\theta^{*}$$", "w": 300},
         ]),

    # ---------------- CF-08 零空间抗恢复 ----------------
    base("CF-08", "CF-08_零空间抗恢复投影",
         "CF-08 零空间 / 抗恢复投影 ΔA_null（机制流程）",
         delta=r"$$\Delta A_{\text{null},\ell,p}=P_{\perp}\Delta A_{\ell,p},\quad P_{\perp}=I-U_{\text{top}}U_{\text{top}}^{\top}$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{null},\ell,p}}\right]\right)$$",
         detail=[
             {"h": "① retain 梯度 top-PC", "f": r"$$G_{\text{retain}}=\big[\nabla_\theta\mathcal L_R^{(1)}\ \cdots\ \nabla_\theta\mathcal L_R^{(m)}\big],\quad U_{\text{top}}=\mathrm{top\!-\!r\ PC}(G_{\text{retain}}),\quad P_{\perp}=I-U_{\text{top}}U_{\text{top}}^{\top}$$", "w": 360},
             {"h": "② 零空间投影 ΔA", "f": r"$$\Delta A_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,\ell}(p),\quad \Delta A_{\text{null},\ell,p}=P_{\perp}\Delta A_{\ell,p},\quad p\in S_f$$", "w": 300},
             {"h": "③ 与中性约束联动", "f": r"$$\mathcal K=\{h: r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\}$$$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{null},\ell,p}}]\right)$$", "w": 320},
             {"h": "④ 参数内化（无 hook）", "f": r"$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L},\quad \theta_0\ \text{frozen}$$$$\text{输出}\ \theta^{*}\ \text{须在移除 hook 后仍保持效应}$$", "w": 300},
             {"h": "⑤ 抗恢复（构造期约束）", "f": r"$$\text{P2/P3/INT4/INT8/Probe 恢复曲线下面积：}\ \Delta A_{\text{null}}\ \text{vs 原始}\ \Delta A$$$$\text{至少两类独立抗恢复证据改善}\Rightarrow\text{受限「深度遗忘」}$$", "w": 320},
         ]),

    # ---------------- CF-09 结构化对照族残差化 ----------------
    base("CF-09", "CF-09_结构化对照族残差化",
         "CF-09 结构化对照族残差化 ΔA_resid（机制流程）",
         donor="结构化对照族 C", donor_f=r"$$\mathcal C=\{X^{\text{rel-swap}},X^{\text{para}},X^{\text{subword}},X^{\text{freq}}\}$$",
         t_minus=r"$$\Delta A^{(c)}_{\ell,p}=A^{+}_{T,\ell}(p)-A^{(c)}_{T,\ell}(p),\ c\in\mathcal C$$",
         delta=r"$$\Delta A_{\text{resid},\ell,p}=(I-\Pi_N)\Delta A_{\ell,p}$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\!\left[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{resid},\ell,p}}\right]\right)$$",
         detail=[
             {"h": "① 对照族伪配对", "f": r"$$\mathcal C=\{X^{\text{rel-swap}},X^{\text{para}},X^{\text{subword}},X^{\text{freq}}\},\quad \Delta A^{(c)}_{\ell,p}=A^{+}_{T,\ell}(p)-A^{(c)}_{T,\ell}(p),\ c\in\mathcal C$$$$\text{不拼}y^{-},\ \text{不算}\ \mathrm{logit}(y^{+})-\mathrm{logit}(y^{-})$$", "w": 360},
             {"h": "② 噪声基", "f": r"$$N_{\ell,p}=\mathrm{span}\{\Delta A^{(c)}_{\ell,p}:c\in\mathcal C\},\quad \Pi_N=B_{\text{null}}B_{\text{null}}^{\top},\quad B_{\text{null}}=\mathrm{orth}(\mathcal C)$$", "w": 300},
             {"h": "③ 残差化", "f": r"$$\Delta A_{\ell,p}=A^{+}_{T,\ell}(p)-A^{-}_{T,\ell}(p),\quad \Delta A_{\text{resid},\ell,p}=(I-\Pi_N)\Delta A_{\ell,p},\quad p\in S_f$$", "w": 300},
             {"h": "④ 目标（非 donor）", "f": r"$$\mathcal K=\{h: r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\}$$$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A_{\text{resid},\ell,p}}]\right)$$", "w": 320},
             {"h": "⑤ 损失 / 反传 / 输出", "f": r"$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \nabla_{\theta}\mathcal{L}\ \text{仅}\ \theta,\quad \lambda_{KL}^{(\mathrm{ablation})}=0,\quad \Rightarrow\theta^{*}$$", "w": 300},
         ]),

    # ---------------- CF-10 纯度驱动渐进 ----------------
    base("CF-10", "CF-10_纯度驱动渐进schedule",
         "CF-10 ΔA 纯度驱动渐进 Λ(k)/ρ_ℓ(k)（机制流程）",
         delta=r"$$\mathrm{pur}_{\ell,p}=\mathrm{AUC}_f(\ell,p)-\mathrm{AUC}_{\text{id}}(\ell,p)\ \text{或}\ w_{\ell,p}$$",
         target=r"$$T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A^{\text{pure}}_{\ell,p}}]\right)$$",
         lossF=r"$$\mathcal{L}_F(k)=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\operatorname{dist}\!\left(h_{S,\ell}(p),T_{\ell,p}\right)$$",
         detail=[
             {"h": "① 纯度谱", "f": r"$$\mathrm{pur}_{\ell,p}=\mathrm{AUC}_f(\ell,p)-\mathrm{AUC}_{\text{id}}(\ell,p)\ \text{或}\ w_{\ell,p},\quad \mathrm{pur}_{\ell}=\frac{1}{|S_f|}\sum_{p\in S_f}\mathrm{pur}_{\ell,p},\quad \mathrm{std}_{\text{seed}}(\mathrm{pur}_{\ell})<\sigma_{\text{pur}}$$", "w": 360},
             {"h": "② 数据驱动 Λ(k)", "f": r"$$\Lambda(k)=\left\{\ell:\ \mathrm{rank}(\mathrm{pur}_{\ell})\le n_k\right\},\quad n_k=\left\lceil|\mathcal L|\cdot\frac{k}{K}\right\rceil$$", "w": 280},
             {"h": "③ 层权重 ρ_ℓ(k)", "f": r"$$\rho_\ell(k)=\frac{\exp(\mathrm{pur}_{\ell}/t_\rho)}{\sum_{\ell'\in\Lambda(k)}\exp(\mathrm{pur}_{\ell'}/t_\rho)},\quad S_f\ \text{固定不参与 schedule}$$$$\text{避最浅/末端 20% 层为软先验}$$", "w": 310},
             {"h": "④ 目标（中性约束）", "f": r"$$\mathcal K=\{h: r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e\},\quad T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}[h^{+}_{\ell}(p)-\alpha\widehat{\Delta A^{\text{pure}}_{\ell,p}}]\right)$$", "w": 330},
             {"h": "⑤ 损失 / 反传 / 输出", "f": r"$$\mathcal{L}_F(k)=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\operatorname{dist}\!\left(h_{S,\ell}(p),T_{\ell,p}\right)$$$$\mathcal{L}=\lambda_F\mathcal{L}_F+\beta\mathcal{L}_R,\quad \nabla_{\theta}\mathcal{L}\Rightarrow\theta^{*}$$", "w": 340},
         ]),
]
