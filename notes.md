## 1 ML³ (meta-learning via learned loss) — core ideas & mathematics

### 1.1 Motivation

Traditional supervised learning fixes a handcrafted loss (e.g. cross-entropy) and only optimises the model parameters θ.
ML³ (Bechtle et al., 2021) instead *meta-learns* a parametric loss network ψ so that optimising θ with that learned loss produces lower *task* loss on a held-out target metric.  Intuitively, ψ becomes a trainable *critic* whose shape guides θ toward better generalisation.

### 1.2 Bilevel optimisation formulation

$$
\min_{\psi}\; 
\mathbb{E}_{(x,y)\sim\mathcal D_{\text{val}}}\;
\mathcal L_{\text{task}}\!\bigl(y,f_{\theta^\*}(x)\bigr)
\quad\text{s.t.}\quad 
\theta^\*=\arg\min_{\theta}\;
\mathbb{E}_{(x,y)\sim\mathcal D_{\text{train}}}\;
\mathcal L_{\psi}\!\bigl(y,f_\theta(x)\bigr)
$$

* **Inner loop**: one or more gradient steps on θ using **learned‐loss**
  $\theta_{t+1}=θ_{t}-\alpha\nabla_{θ}\mathcal L_{\psi}$.
* **Outer loop**: differentiate the validation task loss w\.r.t. ψ.

### 1.3 Loss‐network architecture in ML³

A tiny MLP, applied **output-wise**:

$$
\mathcal L_{\psi}(y,\hat y)=
\frac1C\sum_{c=1}^C
\ell_\psi\bigl(y_{c},\hat y_{c}\bigr)
$$

Original paper used ReLU–Softplus stack; later work showed smoother, *unbounded* activations (e.g. **smooth-leaky-ReLU**) avoid flat regions and permit gradient flow at large error magnitudes.

### 1.4 Practical recipe

1. Randomly initialise θ
2. **Repeat**

   * Randomly initialise ψ.
   * Repeat K times
        * Pull one mini-batch; take a single SGD step on θ using learned loss ψ.
        * Measure task loss on a *different* batch; backprop through the unroll into ψ; update ψ with Adam.

That yields ψ that is useful *only near* the start of training (short-horizon bias) and does **not** adapt online.

### 1.5 Example Implementation

```python
def meta_train_shaped_sine(n_outer_iter,shaped,num_task,n_inner_iter,sine_model,ml3_loss,task_loss_fn, exp_folder):
    theta_ranges = []
    landscape_with_extra = []
    landscape_mse = []

    meta_opt = torch.optim.Adam(ml3_loss.parameters(), lr=ml3_loss.learning_rate)

    for outer_i in range(n_outer_iter):
        # set gradient with respect to meta loss parameters to 0
        batch_inputs, batch_labels, batch_thetas = generate_sinusoid_batch(num_task, 64, n_inner_iter)
        for task in range(num_task):
            sine_model = ShapedSineModel()
            inner_opt = torch.optim.SGD([sine_model.freq], lr=sine_model.learning_rate)
            for step in range(n_inner_iter):
                inputs = torch.Tensor(batch_inputs[task, step, :])
                labels = torch.Tensor(batch_labels[task, step, :])
                label_thetas = torch.Tensor(batch_thetas[task, step, :])

                ''' Updating the frequency parameters, taking gradient of theta wrt meta loss '''
                with higher.innerloop_ctx(sine_model, inner_opt) as (fmodel, diffopt):
                    # use current meta loss to update model
                    yp = fmodel(inputs)
                    meta_input = torch.cat([inputs, yp, labels], dim=1)

                    meta_out = ml3_loss(meta_input)
                    loss = meta_out.mean()
                    diffopt.step(loss)

                    yp = fmodel(inputs)
                    task_loss = task_loss_fn(inputs, yp, labels, shaped, fmodel.freq, label_thetas)

                sine_model.freq = torch.nn.Parameter(fmodel.freq.clone().detach())
                inner_opt = torch.optim.SGD([sine_model.freq], lr=sine_model.learning_rate)

                ''' updating the learned loss '''
                meta_opt.zero_grad()
                task_loss.mean().backward()
                meta_opt.step()

        if outer_i % 100 == 0:
            print("task loss: {}".format(task_loss.mean().item()))

        torch.save(ml3_loss.state_dict(), f'{exp_folder}/ml3_loss_shaped_sine_' + str(shaped) + '.pt')

        if outer_i%10==0:
            t_range, l_with_extra, l_mse = plot_loss(shaped, exp_folder)
            theta_ranges.append(t_range)
            landscape_with_extra.append(l_with_extra)
            landscape_mse.append(l_mse)
    np.save(f'{exp_folder}/theta_ranges_'+str(shaped)+'_.npy', theta_ranges)
    np.save(f'{exp_folder}/landscape_with_extra_'+str(shaped)+'_.npy',landscape_with_extra)
    np.save(f'{exp_folder}/landscape_mse_'+str(shaped)+'_.npy',landscape_mse)
```

The above is the procedure for training the meta-loss model. At "test" time, the outer meta model is frozen and the inner model is updated based upon the predicted losses from the frozen loss-provider model.

### 1.6 Algorithm description:

Relevant excerpt from paper:

```latex
\begin{algorithm}[H]
\begin{algorithmic}[1]
\footnotesize{
\STATE{$\phi \gets$ randomly initialize}
\WHILE{not done}
\STATE{$\theta \gets$ randomly initialize}
\STATE{$x, y \gets \text{Sample task samples from } \mathcal{T}$}
\STATE{$\learnedLoss = \mathcal{M}(y, \ftheta{x})$}
\STATE{$\theta_\text{new} \gets \theta - \alpha \nabla_{\theta} \mathop{\mathbb{E}}_{x}\left[\learnedLoss \right]$}
\STATE{$\phi \gets \phi - \eta \nabla_\phi \taskLoss(y, f_{\theta_\text{new}})$}
\ENDWHILE
}
\end{algorithmic}
\caption{ML$^3$ at (\textit{meta-train})}
\label{algo:ml3_train_sup}
\end{algorithm}

\vspace{-0.8cm}

\begin{algorithm}[H]
\begin{algorithmic}[1]

\footnotesize{
\STATE{$M \gets \text{\# of optimizee updates}$}
\STATE{$\theta \gets $ randomly initialize}
\FOR{$j \in \{0,\dots,M\}$}
\STATE{$x, y \gets \text{Sample task samples from } \mathcal{T}$}
\STATE{$\learnedLoss = \mathcal{M}(y, \ftheta{x})$}
\STATE{$\theta \gets \theta - \alpha \nabla_{\theta} \mathop{\mathbb{E}}_{x}\left[\learnedLoss \right]$}
\ENDFOR
}
\end{algorithmic}
\caption{ML$^3$ at (\textit{meta-test})}
\label{algo:ml3_test_sup}
\end{algorithm}
```

---

## 2 AdaLFL (Adaptive Loss-Function Learning)

### 2.1 Key innovation

AdaLFL removes the “offline” separation between meta-training and model training.  Instead, θ and ψ are updated **in lock-step throughout the entire training run**:

$$
\begin{aligned}
\theta_{t+1}&=\theta_t-\alpha\nabla_\theta \mathcal L_{\psi_t}\\
\psi_{t+1}&=\psi_t-\eta \nabla_\psi
\;\mathcal L_{\text{task}}\!\bigl(f_{\theta_{t+1}}\bigr)
\end{aligned}
$$

Thus ψ becomes *adaptive*—its shape changes across epochs, often starting steep (fast convergence) and flattening later (implicit LR warm-down / regularisation).

### 2.2 Implementation

Paper Extract:

```latex
\subsection{Loss Function Representation}

In AdaLFL, the choice of loss function parameterization is a small feedforward neural network, which is chosen due to its high expressiveness and design flexibility. Our meta-learned loss function parameterization inspired by \cite{bechtle2021meta,psaros2022meta} is a small feedforward neural network denoted by $\ell_{\phi}$ with two hidden layers and 40 hidden units each, which is applied output-wise (making it invariant to the number of outputs).
%
\begin{equation}
\textstyle \MetaLoss_{\phi}(y, f_{\theta}(x)) = \frac{1}{\mathcal{C}}\sum_{i=0}^{\mathcal{C}}\ell_{\phi}(y_{i}, f_{\theta}(x)_i)
\label{eq:loss-definition}
\end{equation}
%
Crucially, key design decisions are made regarding the activation functions used in $\ell_{\phi}$ to enforce desirable behavior. In \cite{bechtle2021meta}, ReLU activations are used in the hidden layers, and the smooth Softplus activation is used in the output layer to constrain the loss to be non-negative, \textit{i.e.}, $\ell{\phi}:\mathbb{R}^2 \rightarrow \mathbb{R}_{0}^{+}$. Unfortunately, this network architecture is prone to \textit{unintentionally} encouraging overly flat loss functions, see Appendix B.1. Generally, flat regions in the loss function are very detrimental to training as uniform loss is given to non-uniform errors. Removal of the Softplus activation in the output can partially resolve this flatness issue; however, without it, the learned loss functions would violate the typical constraint that a loss function should be at least $\mathcal{C}^1$, \textit{i.e.}, continuous in the zeroth and first derivatives. 

Alternative smooth activations, such as Sigmoid, TanH, ELU, etc., can be used instead; however, due to their range-bounded limits, they are also prone to encouraging loss functions that have large flat regions when their activations saturate. Therefore, to inhibit this behavior, the unbounded leaky ReLU \cite{maas2013rectifier} is combined with the smooth ReLU, \textit{i.e.}, SoftPlus \cite{dugas2000incorporating}, as follows:
%
\begin{equation}
\varphi_{hidden}(x) = \tfrac{1}{\beta} \log(e^{\beta x} + 1) \cdot (1 - \gamma) + \gamma x 
\end{equation}
%
This \textit{smooth leaky ReLU} activation function with leak parameter $\gamma$ and smoothness parameter $\beta$ has desirable characteristics for representing a loss function. It is smooth and has linear asymptotic behavior necessary for tasks such as regression, where extrapolation of the learned loss function can often occur. Furthermore, as its output is not bounded when $\gamma > 0$, it does not encourage flatness in the learned loss function. See Appendix B.2 and D.3 for more details.

\setlength{\textfloatsep}{3.5mm}
\begin{algorithm}[t!]

\caption{Loss Function Initialization (Offline)}
\label{algorithm:offline}

\vspace{2mm}
\setlength\parindent{10pt}
\textbf{Input:} $\Loss_{\Task} \leftarrow$ Task loss function (meta-objective)
\vspace{-1mm}

\hrulefill

\begin{algorithmic}[1]

    \STATE $\MetaLoss_{\phi_{0}} \leftarrow$ Initialize parameters of meta learner
    
    \FOR{$i \in \{0, ... , \mathcal{S}_{init}\}$}

        \STATE $\theta_{0} \leftarrow$ Reset parameters of base learner

        \FOR{$j \in \{0, ... , \mathcal{S}_{inner}\}$}
            \STATE $X$, $y$ $\leftarrow$ Sample from $\Dataset_{train}$
            \STATE $\MetaLoss_{learned} \leftarrow \MetaLoss_{\phi_{i}}(y, f_{\theta_{j}}(X))$
            \STATE $\theta_{j + 1} \leftarrow \theta_{j} - \alpha \nabla_{\theta_{j}} \MetaLoss_{learned}$
        \ENDFOR
        
        \STATE $X$, $y$ $\leftarrow$ Sample from $\Dataset_{valid}$
        \STATE $\Loss_{task} \leftarrow \Loss_{\Task}(y, f_{\theta_{j + 1}}(X))$
        \STATE $\phi_{i+1} \leftarrow \phi_{i} - \eta \nabla_{\phi_{i}}\Loss_{task}$

    \ENDFOR

\end{algorithmic}
\end{algorithm}

\subsection{Loss Function Initialization}

One challenge for online loss function learning is achieving a stable and performant initial set of parameters for the learned loss function. If $\phi$ is initialized poorly, too much time is spent on fixing $\phi$ in the early stages of the learning process, resulting in poor base convergence, or in the worst case, $f_{\theta}$ to diverge. To address this, offline loss function learning using \textit{Meta-learning via Learned Loss} (ML$^3$) \cite{bechtle2021meta} is utilized to fine-tune the initial loss function to the base model prior to online loss function learning. The initialization process is summarized in Algorithm \ref{algorithm:offline}, where $\mathcal{S}_{init}=2500$. In AdaLFL's initialization process one step on $\theta$ is taken for each step in $\phi$, \textit{i.e.}, inner gradient steps $\mathcal{S}_{inner}=1$. However, if $\mathcal{S}_{inner} < 1$, implicit differentiation \cite{lorraine2020optimizing,gao2022loss} can instead be utilized to reduce the initialization process's memory and computational overhead.

\subsection{Online Meta-Optimization}

\begin{algorithm}[t!]

\caption{Loss Function Adaptation (Online)}
\label{algorithm:online}

\vspace{2mm}
\setlength\parindent{10pt}

\textbf{Input:} $\MetaLoss_{\phi} \leftarrow$ Learned loss function (base-objective)

\textbf{Input:} $\Loss_{\Task} \leftarrow$ Task loss function (meta-objective)

\vspace{-1mm}

\hrulefill

\begin{algorithmic}[1]

    \STATE $\theta_{0} \leftarrow$ Initialize parameters of base learner
    
    \FOR{$i \in \{0, ... , \mathcal{S}_{train}\}$}

        \STATE $X$, $y$ $\leftarrow$ Sample from $\Dataset_{train}$
        \STATE $\MetaLoss_{learned} \leftarrow \MetaLoss_{\phi_{i}}(y, f_{\theta_{i}}(X))$
        \STATE $\theta_{i+1} \leftarrow \theta_{i} - \alpha \nabla_{\theta_{i}} \MetaLoss_{learned}$
        \STATE $X$, $y$ $\leftarrow$ Sample from $\Dataset_{valid}$
        \STATE $\Loss_{task} \leftarrow \Loss_{\Task}(y, f_{\theta_{i+1}}(X))$
        \STATE $\phi_{i+1} \leftarrow \phi_{i} - \eta \nabla_{\phi_{i}}\Loss_{task}$

    \ENDFOR

\end{algorithmic}

\end{algorithm}

\begin{figure*}[h!]
    
    \centering
    \includegraphics[width=1\textwidth]{figures/computational-graph-2.pdf}
    \captionsetup{justification=centering}
    \captionsetup{belowskip=-3mm}    
    \caption{Computational graph of AdaLFL, where $\theta$ is updated using $\MetaLoss_{\phi}$ in the inner loop (\textit{Base Update}). The optimization path is tracked in the computational graph and then used to update $\phi$ based on the meta-objective in the outer loop (\textit{Meta Update}). The dashed lines show the gradients for $\theta$ and $\phi$ with respect to their given objectives.}

\label{fig:computational-graph}
\end{figure*}

\noindent
To optimize $\phi$, unrolled differentiation is utilized in the outer loop to update the learned loss function after each update to the base model parameters $\theta$ in the inner loop, which occurs via vanilla backpropagation. This is conceptually the simplest way to optimize $\phi$ as all the intermediate iterates generated by the optimizer in the inner loop can be stored and then backpropagate through in the outer loop \cite{maclaurin2015gradient}. The full iterative learning process is summarized in Algorithm \ref{algorithm:online} and proceeds as follows: perform a forward pass $f_{\theta_{t}}(x)$ to obtain an initial set of predictions. The learned loss function $\MetaLoss_{\phi}$ is then used to produce a base loss value
%
\begin{equation}
\MetaLoss_{learned} = \MetaLoss_{\phi_{t}}(y, f_{\theta_{t}}(x)).
\label{eq:loss-base}
\end{equation}
%
Using $\MetaLoss_{learned}$, the current weights $\theta_{t}$ are updated by taking a step in the opposite direction of the gradient of the loss with respect to $\theta_{t}$, where $\alpha$ is the base learning rate. 
%
\begin{equation}
\begin{split}
\theta_{t+1}
& = \theta_{t} - \alpha \nabla_{\theta_{t}} \MetaLoss_{\phi_{t}}(y, f_{\theta_{t}}(x)) \\
& = \theta_{t} - \alpha \nabla_{\theta_{t}} \mathbb{E}_{X, y} \big[ \MetaLoss_{\phi_{t}}(y, f_{\theta_{t}}(x)) \big]
\end{split}
\label{eq:backward-base}
\end{equation}
%
which can be further decomposed via the chain rule as shown in Equation (\ref{eq:backward-base-decompose}). Importantly, all the intermediate iterates generated by the (base) optimizer at the $t^{th}$ time-step when updating $\theta$ are stored in memory. 
%
\begin{equation}
\theta_{t+1} = \theta_{t} - \alpha \nabla_{f} \MetaLoss_{\phi_{t}}(y, f_{\theta_{t}}(x)) \nabla_{\theta_{t}}f_{\theta_{t}}(x)
\label{eq:backward-base-decompose}
\end{equation}
%
$\phi_{t}$ can now be updated to $\phi_{t+1}$ based on the learning progression made by $\theta$. Using $\theta_{t+1}$ as a function of $\phi_{t}$, compute a forward pass using the updated base weights $f_{\theta_{t+1}}(x)$ to obtain a new set of predictions. The instances can either be sampled from the training set or a held-out validation set. The new set of predictions is used to compute the \textit{task loss} $\Loss_{\Task}$ to optimize $\phi_{t}$ through $\theta_{t+1}$
%
\begin{equation}
\Loss_{task} = \Loss_{\Task}(y, f_{\theta_{t+1}}(x))
\label{eq:loss-meta}
\end{equation}
%
where $\Loss_{\Task}$ is selected based on the respective application. For example, the squared error loss for the task of regression or the cross-entropy loss for classification. The task loss is a crucial component for embedding the end goal task into the learned loss function. Optimization of the current meta-loss network loss weights $\phi_{t}$ now occurs by taking the gradient of $\Loss_{\Task}$, where $\eta$ is the meta learning rate.
%
\begin{equation}
\begin{split}
\phi_{t+1}
& = \phi_{t} - \eta \nabla_{\phi_{t}}\Loss_{\Task}(y, f_{\theta_{t+1}}(x)) \\
& = \phi_{t} - \eta \nabla_{\phi_{t}} \mathbb{E}_{X, y} \big[ \Loss_{\Task}(y, f_{\theta_{t+1}}(x)) \big]
\end{split}
\label{eq:backward-meta}
\end{equation}
%
where the gradient computation is decomposed by applying the chain rule as shown in Equation (\ref{eq:backward-meta-decompose}) where the gradient with respect to the meta-loss network weights $\phi_{t}$ requires the updated model parameters $\theta_{t+1}$ from Equation (\ref{eq:backward-base}). 
%
\small
\begin{align}
\phi_{t+1} 
&= \phi_{t} - \eta \nabla_{f}\Loss_{\Task} \nabla_{\theta_{t+1}} f_{\theta_{t+1}} \nabla_{\phi_{t}}\theta_{t+1}(\phi_{t}) \\
&= \phi_{t} - \eta \nabla_{f}\Loss_{\Task} \nabla_{\theta_{t+1}} f_{\theta_{t+1}} \nabla_{\phi_{t}}[\theta_{t} - \alpha \nabla_{\theta_{t}}\MetaLoss_{\phi_{t}}]
\label{eq:backward-meta-decompose}
\end{align}
\normalsize
%
This process is repeated for a fixed number of gradient steps $S_{train}$, which is identical to what would typically be used for training $f_{\theta}$. An overview and summary of the full associated data flow between the inner and outer optimization of $\theta$ and $\phi$, respectively, is given in Figure \ref{fig:computational-graph}.
```

### 2.3 Observed emergent behaviours

* **Implicit LR scheduling**: scaling learned loss is equivalent to changing effective α.
* **Early-stopping regularisation**: ψ often becomes nearly flat late in training, halting updates on outliers.
* Short-term bias largely removed; learned loss keeps guiding θ for tens of thousands of steps.

---

## 3 Single homogeneous model as both language model **and** loss predictor

### 3.1 Your proposal

Remove the separate critic network.  Let the *same* Transformer weights serve dual roles:

1. Standard autoregressive language-modeling. (ie `input -> prediction`)
2. Loss prediction mode triggered by special [LOSS] token, such that `loss_input = input ⊕ prediction ⊕ [LOSS]`, outputs a next-token token prediction for [LOSS] that we can take the mean of to get a scaler, which we then backpropogate with respect to.

Training protocol we tried:

* **Phase 1 (meta-train)**

  * Inner loop: get model preds on input, get predicted loss via `input ⊕ prediction ⊕ [LOSS]`, backpropogate on predicted loss.
  * Outer loop: get model preds on (a potentially different) input, get CE loss, backpropogate end-to-end.
  * **Interpolation** of inner θ′ back into outer θ to blend task knowledge and critic knowledge.

* **Phase 2 (self-loss-only)**

  * continue SGD on θ driven solely by its own learned-loss output; record true CE for diagnostic to see if we can continue learning purely with self-produced loss.

### 3.2 Why this might work

* Could evolve into a **Gödel-machine-style** self-improver: code that rewrites itself guided by an internal theorem prover (here: the learned loss).
* Learned loss could incorporate more semantically nuanced information than pure CE based loss
* Self-producing loss models might be able to be adapted to work in post-training in circumstances that don't have cleanly labeled target values but instead have more ambiguous environmental feedback, since the meta-learned loss-production procedures might be general enough to be adapted to less ideal conditions (analogously to how pretrained LLMs can be finetuned into instruct models.)

### 3.3 The pathology we observed

During Phase 2, the self-loss keeps decreasing (to −0.5, −0.6 …) but true CE plateaus around 3.4 → **reward hacking**.
Because the same weights generate predictions *and* evaluate them, the easiest way to lower predicted loss is to *reshape the loss head* rather than improve language accuracy.

---

## 4 Diagnosing the reward-hacking failure

1. **Under-constrained output space** — loss scalar can freely shift/scale; with shared parameters the easiest solution is to move that bias rather than reduce language errors.
2. **Parameter coupling** — update meant for LM weights also changes loss head because they are identical matrices.
3. **No Lipschitz / monotonicity guarantees** — predicted loss need not be ≥0 or correlate with CE.

---

## 5 Mitigation strategies discussed

### 5.1 Architectural separation

* **Disjoint upper layers**: share lower layers, but have disjoint later layers with their own weights.  Gradient routing: inner loop updates all lower weights plus LM specific later weights; outer loop updates all lower weights plus loss specific later weights.

### 5.2 Constrained loss parameterisation

* Use **smooth-leaky-ReLU** output with positive range, optional monotonicity penalty
  $\ell_\psi(u)=\operatorname{SLReLU}(h_\psi(u))$  ensures decrease ↔ lower error.
* Normalise predicted loss batch-wise: zero-mean, unit-variance → eliminates global bias exploit.

### 5.3 Regularised training schedules

* Periodic **outer CE anchor** even during self phase (e.g. every 20 steps) to nip reward hacking.

### 5.4 First-order or truncated-gradient inner loop

Reducing second-order credit assignment (FO-MAML style) can (paradoxically) help because loss head cannot perfectly back-prop bias to shared weights.