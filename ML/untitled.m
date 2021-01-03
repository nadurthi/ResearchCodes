RGB = imread('/home/na0043/Insync/n.adurthi@gmail.com/Google Drive/repos/SLAM/ML/img2.png');
I  = rgb2gray(RGB);
BW = edge(I,'canny');
[H,T,R] = hough(BW,'Theta',-80:5:80);

figure
imshow(imadjust(rescale(H)),'XData',T,'YData',R,...
      'InitialMagnification','fit');
title('Hough transform of gantrycrane.png');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(gca,hot);