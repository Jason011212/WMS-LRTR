function [E] = solve_l1l1l2(X,lambda) %对应论文中公式（17）
[H,W,D] = size(X);
nm=sqrt(sum(X.^2,3));
nms=max(nm-ones(H,W)*lambda,0);
sw=repmat(nms./nm,[1,1,D]);
E=sw.*X;
end