**Result:** Systems of linear equations relating **p*** and **p** to **q**  

Initialize \(P\) as a \(B \times \aleph_R\) matrix of 0s;  
Initialize \(P^{*}\) as a \(B \times \aleph_R\) matrix of 0s;  
Initialize \(\Lambda\) as a \(B \times B\) matrix of 0s;  

for \(b \in 1, \ldots, B\) do  
&emsp;for \(\gamma \in 1, \ldots, \aleph_R\) do  
&emsp;&emsp;Initialize \(\omega\) as an empty vector of length \(|R| \; ( = n - J)\);  
&emsp;&emsp;for \(i \in R\) do  
&emsp;&emsp;&emsp;Set \(\omega_i := g_{W_i}(w_{b,\mathcal{L}}, r_{\gamma})\);  
&emsp;&emsp;end  
&emsp;&emsp;if \(\omega = w_{b,\mathcal{R}}\) then  
&emsp;&emsp;&emsp;\(P_{b,\gamma} := 1\);  
&emsp;&emsp;&emsp;\(\Lambda_{b,b} := p\{ W_{\mathcal{L}} = w_{b,\mathcal{L}} \}\);  
&emsp;&emsp;&emsp;\(P^{*}_{b,\gamma} := p\{ W_{\mathcal{L}} = w_{b,\mathcal{L}} \}\);  
&emsp;&emsp;end  
&emsp;end  
end  

**Algorithm 1:** An algorithm to determine a system of linear equations relating **p** and **p*** to **q**.

