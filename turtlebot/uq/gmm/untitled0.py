# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:13:17 2020

@author: Nagnanamus
"""
import numpy as np
import os
import pickle as pkl
D={'rule1':{},'rule2':{},'rule3':{}}

ss="""
3 0.57735026918962573 -1.078462714638291 0 1.078462714638291 0.24441982292370251 0.51116035415259509 0.24441982292370251 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 0.44721359549995793 -1.6953835499623631 -0.80345438379832324 0 0.80345438379832324 1.6953835499623631 0.075110102179119301 0.24765415661851711 0.35447148240472709 0.24765415661851711 0.075110102179119301 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
7 0.3779644730092272 -2.1073616922654832 -1.3298721132043589 -0.64876246076468813 0 0.64876246076468813 1.3298721132043589 2.1073616922654832 0.028799777829538849 0.10987548613678071 0.22237907516773489 0.27789132173189118 0.22237907516773489 0.10987548613678071 0.028799777829538849 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
9 0.33333333333333331 -2.4187475226854089 -1.7193605089949671 -1.1145293827435361 -0.5492665472161351 0 0.5492665472161351 1.1145293827435361 1.7193605089949671 2.4187475226854089 0.01266710315422743 0.051316824387990019 0.1221638546130597 0.1980911515599085 0.23152213256962881 0.1980911515599085 0.1221638546130597 0.051316824387990019 0.01266710315422743 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
11 0.30151134457776357 -2.670537180265157 -2.0286898728807272 -1.477697482804665 -0.96797072986189392 -0.47937166432707262 0 0.47937166432707262 0.96797072986189392 1.477697482804665 2.0286898728807272 2.670537180265157 0.0061199058304968624 0.025450156637052392 0.066271547802523617 0.1242223152135368 0.17792090373655961 0.2000303415596614 0.17792090373655961 0.1242223152135368 0.066271547802523617 0.025450156637052392 0.0061199058304968624 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
13 0.27735009811261457 -2.8828632184116358 -2.2856744544530709 -1.77543510556555 -1.3064032676537829 -0.86039237194213647 -0.42726555284406859 0 0.42726555284406859 0.86039237194213647 1.3064032676537829 1.77543510556555 2.2856744544530709 2.8828632184116358 0.0031678498118064549 0.01330943531766905 0.03658870125902456 0.075060131904974889 0.1218887297008967 0.16145768856845591 0.1770549268743449 0.16145768856845591 0.1218887297008967 0.075060131904974889 0.03658870125902456 0.01330943531766905 0.0031678498118064549 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
15 0.2581988897471611 -3.0670828831023771 -2.5058705448424949 -2.0279264473158132 -1.59049006302482 -1.1767103432954871 -0.77745521783992766 -0.38673757403703718 0 0.38673757403703718 0.77745521783992766 1.1767103432954871 1.59049006302482 2.0279264473158132 2.5058705448424949 3.0670828831023771 0.001729432592805709 0.0072802480530913946 0.02071800060206181 0.045139490477766171 0.079735835020289497 0.1177554527259974 0.14791617469497539 0.15945073166602541 0.14791617469497539 0.1177554527259974 0.079735835020289497 0.045139490477766171 0.02071800060206181 0.0072802480530913946 0.001729432592805709 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
17 0.242535625036333 -3.230232170637648 -2.6988197644550489 -2.2473174848768682 -1.8353874449141641 -1.447189860427331 -1.0742672230485999 -0.71123607270641609 -0.35419853546942109 0 0.35419853546942109 0.71123607270641609 1.0742672230485999 1.447189860427331 1.8353874449141641 2.2473174848768682 2.6988197644550489 3.230232170637648 0.00098524695604325602 0.0041365423776581634 0.012042257404060799 0.027364625413012409 0.051302355459059548 0.081810124139960194 0.1129934596762521 0.1366294940546926 0.14547178903852179 0.1366294940546926 0.1129934596762521 0.081810124139960194 0.051302355459059548 0.027364625413012409 0.012042257404060799 0.0041365423776581634 0.00098524695604325602 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
19 0.2294157338705618 -3.376973806944743 -2.8707825717297788 -2.4414666109169478 -2.0507180969222629 -1.683509993257108 -1.331885606825451 -0.99086572197216705 -0.65694811682096088 -0.32742240983244347 0 0.32742240983244347 0.65694811682096088 0.99086572197216705 1.331885606825451 1.683509993257108 2.0507180969222629 2.4414666109169478 2.8707825717297788 3.376973806944743 0.00058130816276837517 0.0024277310315266578 0.007175411238725656 0.016808175844932091 0.032920789243460681 0.055554702101355012 0.08226617745353329 0.1081403132505975 0.12709260023156299 0.13406558288307549 0.12709260023156299 0.1081403132505975 0.08226617745353329 0.055554702101355012 0.032920789243460681 0.016808175844932091 0.007175411238725656 0.0024277310315266578 0.00058130816276837517 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
21 0.21821789023599239 -3.51055605781402 -3.0260797875923591 -2.6157384777017461 -2.2429629221768721 -1.8934046414793919 -1.5595034840171169 -1.2365772219315021 -0.92139038516408012 -0.61150702168780113 -0.30495055165693968 0 0.30495055165693968 0.61150702168780113 0.92139038516408012 1.2365772219315021 1.5595034840171169 1.8934046414793919 2.2429629221768721 2.6157384777017461 3.0260797875923591 3.51055605781402 0.00035323567332530223 0.001465182522930507 0.0043743126899990943 0.0104792585648706 0.02121528211878811 0.037377283634947932 0.058361077261690028 0.081724505413556303 0.1034404307034264 0.1189309780989635 0.1245569066350046 0.1189309780989635 0.1034404307034264 0.081724505413556303 0.058361077261690028 0.037377283634947932 0.02121528211878811 0.0104792585648706 0.0043743126899990943 0.001465182522930507 0.00035323567332530223 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
23 0.2085144140570748 -3.6333446105410219 -3.1678277879270191 -2.7739653093077088 -2.416701545368841 -2.0822692741780422 -1.7634359696029771 -1.455751905364661 -1.156177090063093 -0.86246288049261255 -0.57283104516866035 -0.285787202412633 0 0.285787202412633 0.57283104516866035 0.86246288049261255 1.156177090063093 1.455751905364661 1.7634359696029771 2.0822692741780422 2.416701545368841 2.7739653093077088 3.1678277879270191 3.6333446105410219 0.0002201180426915819 0.00090600820353477479 0.0027226821918702828 0.0066334444995085852 0.01377720799761468 0.02510223135905806 0.040852232740835062 0.060098912550283483 0.080575637094649571 0.098998669363684275 0.11186674755116641 0.1164922168102067 0.11186674755116641 0.098998669363684275 0.080575637094649571 0.060098912550283483 0.040852232740835062 0.02510223135905806 0.01377720799761468 0.0066334444995085852 0.0027226821918702828 0.00090600820353477479 0.0002201180426915819 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
25 0.20000000000000001 -3.747101882185945 -3.2983315256328498 -2.918961379179164 -2.5752707448950001 -2.253999830183592 -1.948194992722557 -1.6535967422120781 -1.3673170724495629 -1.0872448399142549 -0.81173903744610842 -0.53945311617850367 -0.26922475445736649 0 0.26922475445736649 0.53945311617850367 0.81173903744610842 1.0872448399142549 1.3673170724495629 1.6535967422120781 1.948194992722557 2.253999830183592 2.5752707448950001 2.918961379179164 3.2983315256328498 3.747101882185945 0.00014019623605284579 0.00057233701022068359 0.0017269432209818011 0.0042616680490982764 0.009030801747891044 0.016899669101975628 0.028427526439247171 0.043499949852647532 0.061056109512842169 0.079066800678196894 0.094851243834722043 0.10569001158001751 0.10955348547221309 0.10569001158001751 0.094851243834722043 0.079066800678196894 0.061056109512842169 0.043499949852647532 0.028427526439247171 0.016899669101975628 0.009030801747891044 0.0042616680490982764 0.0017269432209818011 0.00057233701022068359 0.00014019623605284579 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
27 0.19245008972987529 -3.853188630207351 -3.4193554549543288 -3.0528653738060818 -2.72118752327671 -2.4115114094424861 -2.117125921859206 -1.8339322527759701 -1.5591658337383401 -1.2908221566023239 -1.0273615020944269 -0.76754111749742804 -0.51031164071604096 -0.25474808755189188 0 0.25474808755189188 0.51031164071604096 0.76754111749742804 1.0273615020944269 1.2908221566023239 1.5591658337383401 1.8339322527759701 2.117125921859206 2.4115114094424861 2.72118752327671 3.0528653738060818 3.4193554549543288 3.853188630207351 9.1021325392705762e-05 0.00036846567172716299 0.001114285534297967 0.0027768117960210409 0.0059791345627193994 0.011432882506756959 0.019754773073884839 0.031212139575735341 0.045470536852274902 0.061446110959371969 0.077355676078204788 0.091001427003571084 0.1002409315296572 0.1035116070607692 0.1002409315296572 0.091001427003571084 0.077355676078204788 0.061446110959371969 0.045470536852274902 0.031212139575735341 0.019754773073884839 0.011432882506756959 0.0059791345627193994 0.0027768117960210409 0.001114285534297967 0.00036846567172716299 9.1021325392705762e-05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
29 0.18569533817705189 -3.9526707785693409 -3.532272683670985 -3.177332255516566 -2.8563885006510419 -2.5570332389128438 -2.2727697315066169 -1.999638148457799 -1.7349774998048639 -1.4768694890513869 -1.2238529690766839 -0.97476233027713888 -0.72862865821289036 -0.4846150354779758 -0.24197129177601159 0 0.24197129177601159 0.4846150354779758 0.72862865821289036 0.97476233027713888 1.2238529690766839 1.4768694890513869 1.7349774998048639 1.999638148457799 2.2727697315066169 2.5570332389128438 2.8563885006510419 3.177332255516566 3.532272683670985 3.9526707785693409 6.0108408146582857e-05 0.00024126626436692941 0.00073027566093884875 0.0018335351427912899 0.0039991307883397633 0.0077828432146285634 0.01374825921759913 0.022303434711329881 0.033505990804268707 0.046895852859423132 0.061425723972099909 0.075543338824376027 0.087436560239931099 0.095395437318014489 0.098196485147491269 0.095395437318014489 0.087436560239931099 0.075543338824376027 0.061425723972099909 0.046895852859423132 0.033505990804268707 0.022303434711329881 0.01374825921759913 0.0077828432146285634 0.0039991307883397633 0.0018335351427912899 0.00073027566093884875 0.00024126626436692941 6.0108408146582857e-05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
31 0.17960530202677491 -4.0464048218402926 -3.6381783670823431 -3.2936743159738509 -2.9824008761281982 -2.692313375545877 -2.4171069170720001 -2.1529440148207031 -1.8972527471488769 -1.6481867846610301 -1.404348368622798 -1.1646318932380899 -0.92812873517139671 -0.69406558991005485 -0.4617621398632652 -0.2306002412477732 0 0.2306002412477732 0.4617621398632652 0.69406558991005485 0.92812873517139671 1.1646318932380899 1.404348368622798 1.6481867846610301 1.8972527471488769 2.1529440148207031 2.4171069170720001 2.692313375545877 2.9824008761281982 3.2936743159738509 3.6381783670823431 4.0464048218402926 4.0302277735567802e-05 0.00016040314393191461 0.00048546508378054801 0.001225867732904173 0.00270174917041621 0.0053350584299923388 0.0095992093882985453 0.015920101421927599 0.024539148909161319 0.03536881552788268 0.047885959417226093 0.061109781966770159 0.073694794340477557 0.08413732164000555 0.091056307036583989 0.093479429025811522 0.091056307036583989 0.08413732164000555 0.073694794340477557 0.061109781966770159 0.047885959417226093 0.03536881552788268 0.024539148909161319 0.015920101421927599 0.0095992093882985453 0.0053350584299923388 0.00270174917041621 0.001225867732904173 0.00048546508378054801 0.00016040314393191461 4.0302277735567802e-05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
33 0.1740776559556978 -4.1350865631365661 -3.7379568385751152 -3.4029461796724392 -3.1004464616533589 -2.8187439372852729 -2.551707102673328 -2.2956076662088201 -2.0479517569004679 -1.806954627708891 -1.5712712545319181 -1.33984442625638 -1.111812600207567 -0.88645055908357129 -0.66312912054867335 -0.44128636572360008 -0.22040598518779239 0 0.22040598518779239 0.44128636572360008 0.66312912054867335 0.88645055908357129 1.111812600207567 1.33984442625638 1.5712712545319181 1.806954627708891 2.0479517569004679 2.2956076662088201 2.551707102673328 2.8187439372852729 3.1004464616533589 3.4029461796724392 3.7379568385751152 4.1350865631365661 2.739506347739853e-05 0.0001081237406814651 0.00032695765525287118 0.0008291964850578289 0.001843087332684861 0.0036838663394350331 0.0067315288721110063 0.011374134301439871 0.017917416503535781 0.026474301296487569 0.03686062276633098 0.048530563445733393 0.060582524165541281 0.071851662852482734 0.081082043771487977 0.087146118884289928 0.089260913047940038 0.087146118884289928 0.081082043771487977 0.071851662852482734 0.060582524165541281 0.048530563445733393 0.03686062276633098 0.026474301296487569 0.017917416503535781 0.011374134301439871 0.0067315288721110063 0.0036838663394350331 0.001843087332684861 0.0008291964850578289 0.00032695765525287118 0.0001081237406814651 2.739506347739853e-05 0 0 0 0 0 0 0 0 0 0 0 0
35 0.1690308509457033 -4.219289955599943 -3.832333267767547 -3.5060087630803851 -3.2115185299130089 -2.937452006728448 -2.6778356840180328 -2.4290397984819889 -2.188639894463551 -1.9549049554628219 -1.7265349118069451 -1.502512738872523 -1.2820149103687379 -1.0643539418104091 -0.84893964487895435 -0.6352517822867757 -0.42281988188411102 -0.2112076082828549 0 0.2112076082828549 0.42281988188411102 0.6352517822867757 0.84893964487895435 1.0643539418104091 1.2820149103687379 1.502512738872523 1.7265349118069451 1.9549049554628219 2.188639894463551 2.4290397984819889 2.6778356840180328 2.937452006728448 3.2115185299130089 3.5060087630803851 3.832333267767547 4.219289955599943 1.8854257189034388e-05 7.3805005296146214e-05 0.00022285919000872439 0.0005670201850402829 0.0012691186838625771 0.0025625553506748401 0.0047443000509555027 0.0081446439250372846 0.013070558980945401 0.019727178634200419 0.02813111289574256 0.038037030727725941 0.048901786058296648 0.05990590646355843 0.070040293209788426 0.078249071331671805 0.083602539815203533 0.085462730469604856 0.083602539815203533 0.078249071331671805 0.070040293209788426 0.05990590646355843 0.048901786058296648 0.038037030727725941 0.02813111289574256 0.019727178634200419 0.013070558980945401 0.0081446439250372846 0.0047443000509555027 0.0025625553506748401 0.0012691186838625771 0.0005670201850402829 0.00022285919000872439 7.3805005296146214e-05 1.8854257189034388e-05 0 0 0 0 0 0 0 0
37 0.16439898730535729 -4.2994942147327446 -3.9219097750344458 -3.6035739298638179 -3.3164353111246729 -3.0493626536719081 -2.7965273862165891 -2.5543894399552149 -2.3205868274098131 -2.093436142179999 -1.871676391930635 -1.6543247196927471 -1.440589150252968 -1.2298127354325239 -1.0214360496802239 -0.81497091645379782 -0.60998124952897859 -0.4060685045287048 -0.20286013896802901 0 0.20286013896802901 0.4060685045287048 0.60998124952897859 0.81497091645379782 1.0214360496802239 1.2298127354325239 1.440589150252968 1.6543247196927471 1.871676391930635 2.093436142179999 2.3205868274098131 2.5543894399552149 2.7965273862165891 3.0493626536719081 3.3164353111246729 3.6035739298638179 3.9219097750344458 4.2994942147327446 1.312402079241742e-05 5.0961793411383091e-05 0.0001535930562599783 0.0003917041050329713 0.00088172314238700263 0.0017956577750817689 0.0033618353037326932 0.0058505287360311551 0.009540288408118892 0.014665058774169969 0.021348224963755141 0.02953605762133156 0.038947444801988203 0.049057221735934159 0.05912564322544258 0.068276968621012554 0.075617870647903124 0.080374834250792906 0.082022518033643035 0.080374834250792906 0.075617870647903124 0.068276968621012554 0.05912564322544258 0.049057221735934159 0.038947444801988203 0.02953605762133156 0.021348224963755141 0.014665058774169969 0.009540288408118892 0.0058505287360311551 0.0033618353037326932 0.0017956577750817689 0.00088172314238700263 0.0003917041050329713 0.0001535930562599783 5.0961793411383091e-05 1.312402079241742e-05 0 0 0 0
39 0.1601281538050871 -4.3761037192531456 -4.0071918122297099 -3.6962369601762051 -3.4158786984101961 -3.1552442811452752 -2.9086388609714748 -2.6726046205980718 -2.4448363296226141 -2.2236932973701289 -2.00794904007385 -1.796650315363159 -1.5890319318694479 -1.384462296769541 -1.1824069542197631 -0.9824031645788438 -0.78404151402467648 -0.58695212436768074 -0.39079392282249781 -0.1952459507632788 0 0.1952459507632788 0.39079392282249781 0.58695212436768074 0.78404151402467648 0.9824031645788438 1.1824069542197631 1.384462296769541 1.5890319318694479 1.796650315363159 2.00794904007385 2.2236932973701289 2.4448363296226141 2.6726046205980718 2.9086388609714748 3.1552442811452752 3.4158786984101961 3.6962369601762051 4.0071918122297099 4.3761037192531456 9.2307196303143448e-06 3.5562753382754079e-05 0.000106944148691712 0.00027318190628093241 0.0006178030082777448 0.0012673350030340299 0.0023955755129757531 0.004218347602164798 0.0069747802679760457 0.010893333876206021 0.016144516607610598 0.022786815645448631 0.03071655327571195 0.039634684717545381 0.049042729404146862 0.058275565208968622 0.066571296896008936 0.073169507700232422 0.077421288817510966 0.078889893856391086 0.077421288817510966 0.073169507700232422 0.066571296896008936 0.058275565208968622 0.049042729404146862 0.039634684717545381 0.03071655327571195 0.022786815645448631 0.016144516607610598 0.010893333876206021 0.0069747802679760457 0.004218347602164798 0.0023955755129757531 0.0012673350030340299 0.0006178030082777448 0.00027318190628093241 0.000106944148691712 3.5562753382754079e-05 9.2307196303143448e-06
""".strip().strip('\n')

L=ss.split('\n')
L = [s for s in L if len(s)>0]

for i in range(len(L)):
    L[i]=[float(g) for g in L[i].split(' ')]
    

for i in range(len(L)):
    D['rule1'][int(L[i][0])]={}
    N = int(L[i][0])
    D['rule1'][int(L[i][0])]['stds'] = [L[i][1]]*N
    D['rule1'][int(L[i][0])]['means'] = L[i][2:2+N]
    D['rule1'][int(L[i][0])]['wts'] = L[i][2+N:2+2*N]
    


ss="""
3 0.66233778214052597 -1.0618454690787467 0 1.0618454690787471 0.22706897340874665 0.54586205318250669 0.22706897340874671 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 0.54687270570421065 -1.7642943565100089 -0.83584574108546661 0 0.83584574108546661 1.7642943565100091 0.054058258084466342 0.2494630002479214 0.39295748333522462 0.2494630002479214 0.054058258084466342 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
7 0.48204572422389641 -2.2683006150001632 -1.4303019352895849 -0.69756445968605163 0 0.69756445968605163 1.4303019352895849 2.2683006150001628 0.015183128558201734 0.092214593039124507 0.2349334396447694 0.31533767751580871 0.2349334396447694 0.092214593039124507 0.01518312855820173 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
9 0.43869133765083079 -2.6664710731199031 -1.893402157422267 -1.2268295168285059 -0.6044888046014083 0 0.6044888046014083 1.2268295168285059 1.893402157422267 2.6664710731199031 0.0047959978792767406 0.033932612846572979 0.111672303375039 0.2160249710482377 0.26714822970174718 0.2160249710482377 0.111672303375039 0.033932612846572979 0.0047959978792767397 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
11 0.40689068074382112 -2.9985802397662367 -2.2749921372229029 -1.6562368084480441 -1.084614480198161 -0.53705812667495512 0 0.53705812667495512 1.084614480198161 1.6562368084480441 2.2749921372229029 2.9985802397662362 0.001652792425963308 0.0128876549496003 0.049799619159453822 0.1201828758008336 0.19856354022906819 0.23382703487016149 0.19856354022906819 0.1201828758008336 0.049799619159453822 0.0128876549496003 0.001652792425963308 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
13 0.3821828923953659 -3.2852841099128414 -2.6010850123025882 -2.0192333067622452 -1.4852860743811129 -0.97800327573619683 -0.48561650998289929 0 0.48561650998289929 0.97800327573619683 1.4852860743811129 2.0192333067622452 2.6010850123025882 3.2852841099128409 0.00060948637153389492 0.005082660065498526 0.02193528709740147 0.061474494935185697 0.12287165854616949 0.18343660871814019 0.2091796085321414 0.18343660871814019 0.12287165854616949 0.061474494935185697 0.02193528709740147 0.005082660065498526 0.00060948637153389492 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
15 0.36221445500931537 -3.5387151770701681 -2.8868942360376408 -2.3347662413766761 -1.8304390355746949 -1.3539027309782401 -0.89438900250347131 -0.4448671853983327 0 0.4448671853983327 0.89438900250347131 1.3539027309782401 1.8304390355746949 2.3347662413766761 2.8868942360376408 3.5387151770701681 0.00023737026441043854 0.0020789434556247308 0.0097277458336924479 0.030380299657740889 0.06954765533057318 0.1224966003265316 0.17048976323224871 0.19008324379835601 0.17048976323224871 0.1224966003265316 0.06954765533057318 0.030380299657740889 0.0097277458336924479 0.0020789434556247308 0.00023737026441043849 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
17 0.3456063031210454 -3.7666274132006241 -3.1420781626967522 -2.614627199687392 -2.1344838018460419 -1.682564147215434 -1.248758668273201 -0.82666255009980794 -0.4116537623482342 0 0.4116537623482342 0.82666255009980794 1.248758668273201 1.682564147215434 2.1344838018460419 2.614627199687392 3.1420781626967522 3.7666274132006241 9.6729227907634796e-05 0.00087928686495330653 0.0043764246666415938 0.01483926686620517 0.037618808255337838 0.074900050416803296 0.1205202365870346 0.15938032804665739 0.17477773813691819 0.15938032804665739 0.1205202365870346 0.074900050416803296 0.037618808255337838 0.01483926686620517 0.0043764246666415938 0.00087928686495330653 9.6729227907634796e-05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
19 0.3314876857226845 -3.9742903586625382 -3.3731537019096409 -2.866660418651596 -2.4068061036592332 -1.975251688750995 -1.5623708363332991 -1.1621705388500889 -0.77045116493280352 -0.38397154160099101 0 0.38397154160099101 0.77045116493280352 1.1621705388500889 1.5623708363332991 1.975251688750995 2.4068061036592332 2.866660418651596 3.3731537019096409 3.9742903586625382 4.0959927785812063e-05 0.0003833170154220446 0.0020030632360121012 0.0072454724724840399 0.019891162354755031 0.043544307436483201 0.078286103434783386 0.1177320240936172 0.1497791756480378 0.1621888287612388 0.1497791756480378 0.1177320240936172 0.078286103434783386 0.043544307436483201 0.019891162354755031 0.0072454724724840399 0.0020030632360121012 0.0003833170154220446 4.0959927785812063e-05 0 0 0 0 0 0 0 0 0 0 0 0
21 0.31927709587277919 -4.1654544975612948 -3.5847270278034729 -3.0963503415603211 -2.6538713043626121 -2.2395752411390801 -1.8442160265834899 -1.4620991216395449 -1.0893054082527469 -0.72289333093494612 -0.36048157078723392 0 0.36048157078723392 0.72289333093494612 1.0893054082527469 1.4620991216395449 1.8442160265834899 2.2395752411390801 2.6538713043626121 3.0963503415603211 3.5847270278034729 4.1654544975612948 1.7928271012608499e-05 0.0001717240176836573 0.00093338455495501173 0.0035571644596835558 0.010411435489102309 0.02458561475688964 0.048260415325347521 0.080273434945703298 0.1145668415375188 0.14141185374865481 0.1516204057868977 0.14141185374865481 0.1145668415375188 0.080273434945703298 0.048260415325347521 0.02458561475688964 0.010411435489102309 0.0035571644596835558 0.00093338455495501173 0.0001717240176836573 1.7928271012608499e-05 0 0 0 0 0 0 0 0
23 0.30856883546708808 -4.3429160342003073 -3.7802042853511502 -3.3077120734860341 -2.8803546598471481 -2.4809555728173591 -2.1005821982193869 -1.7337714306549119 -1.376806958866871 -1.0269490550698159 -0.68203683546110694 -0.34025796321795271 0 0.34025796321795271 0.68203683546110694 1.0269490550698159 1.376806958866871 1.7337714306549119 2.1005821982193869 2.4809555728173591 2.8803546598471481 3.3077120734860341 3.7802042853511502 4.3429160342003073 8.0767978409868113e-06 7.8841476219599724e-05 0.00044267158122080348 0.001761268439452134 0.005433823261384009 0.01365227881398811 0.028784504636507221 0.05193816134410896 0.081270580737745568 0.1112683354774905 0.13406043859943001 0.14260203766922419 0.13406043859943001 0.1112683354774905 0.081270580737745568 0.05193816134410896 0.028784504636507221 0.01365227881398811 0.005433823261384009 0.001761268439452134 0.00044267158122080348 7.8841476219599724e-05 8.0767978409868113e-06 0 0 0 0
25 0.29906975624424409 -4.5087249021078062 -3.962082407346351 -3.503681374280518 -3.089658870201895 -2.7033115520056019 -2.3359738463666502 -1.982370895625891 -1.6389423997015931 -1.303094940303938 -0.97281833742997437 -0.64646682110893416 -0.32262220319392459 0 0.32262220319392459 0.64646682110893416 0.97281833742997437 1.303094940303938 1.6389423997015931 1.982370895625891 2.3359738463666502 2.7033115520056019 3.089658870201895 3.503681374280518 3.962082407346351 4.5087249021078062 3.7336558910098371e-06 3.7019824331238441e-05 0.000213581063590121 0.0008810144269564904 0.0028405557242139568 0.0075155860909479361 0.016814874316551019 0.032449810081489568 0.054757484383982952 0.081568751791387534 0.1079714975410543 0.1275477728036146 0.13479663659197849 0.1275477728036146 0.1079714975410543 0.081568751791387534 0.054757484383982952 0.032449810081489568 0.016814874316551019 0.0075155860909479361 0.0028405557242139568 0.0008810144269564904 0.000213581063590121 3.7019824331238441e-05 3.7336558910098371e-06

"""


L=ss.split('\n')
L = [s for s in L if len(s)>0]

for i in range(len(L)):
    L[i]=[float(g) for g in L[i].split(' ')]
    

for i in range(len(L)):
    D['rule2'][int(L[i][0])]={}
    N = int(L[i][0])
    D['rule2'][int(L[i][0])]['stds'] = [L[i][1]]*N
    D['rule2'][int(L[i][0])]['means'] = L[i][2:2+N]
    D['rule2'][int(L[i][0])]['wts'] = L[i][2+N:2+2*N]



ss="""
3 0.75983568565159254 -0.98690160801704696 0 0.98690160801704685 0.2080748323003781 0.5838503353992438 0.2080748323003781 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 0.66874030497642201 -1.7377631137128715 -0.82366719039403635 0 0.82366719039403635 1.7377631137128711 0.035704218512032669 0.2463986032595161 0.4357943564569024 0.2463986032595161 0.035704218512032669 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
7 0.61478815295126443 -2.3157849286508032 -1.4604041960075189 -0.71220060476374614 0 0.71220060476374614 1.4604041960075189 2.3157849286508032 0.0065588408361459466 0.071358708979421301 0.2432918183943592 0.35758126358014702 0.2432918183943592 0.071358708979421301 0.0065588408361459474 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
9 0.57735026918962584 -2.793037861235109 -1.9830401832079869 -1.284707716562794 -0.63294858181539893 0 0.63294858181539893 1.284707716562794 1.9830401832079869 2.793037861235109 0.0012931188094573907 0.018855743414952379 0.094790678797169597 0.23114008742089881 0.30784074311504372 0.23114008742089881 0.094790678797169597 0.018855743414952379 0.0012931188094573911 0 0 0 0 0 0 0 0 0 0 0 0
11 0.54910048677611245 -3.2038367993790171 -2.430074565016819 -1.768747627209784 -1.158131931153336 -0.57341709159145515 0 0.57341709159145515 1.158131931153336 1.768747627209784 2.430074565016819 3.2038367993790171 0.00027104363087999733 0.004882833979149407 0.032049768088637402 0.10868592198452801 0.21767935736662289 0.27286214990036439 0.21767935736662289 0.10868592198452801 0.032049768088637402 0.004882833979149407 0.00027104363087999728 0 0 0 0 0 0 0 0
13 0.52664038784792655 -3.5670670223655567 -2.8231499172618819 -2.1910384939875058 -1.611379637208078 -1.0609145039377099 -0.52675309315122876 0 0.52675309315122876 1.0609145039377099 1.611379637208078 2.1910384939875058 2.8231499172618819 3.5670670223655572 5.988118930279301e-05 0.001270164459607815 0.01017293838370935 0.043777070480805959 0.11644526852045919 0.20495272817404769 0.24664389758413441 0.20495272817404769 0.11644526852045919 0.043777070480805959 0.01017293838370935 0.001270164459607815 5.988118930279301e-05 0 0 0 0
15 0.50813274815461473 -3.8947232743627107 -3.1759640344395761 -2.5678004898372619 -2.0127475000528441 -1.4885581155719569 -0.98326175895304146 -0.48904985926434918 0 0.48904985926434918 0.98326175895304146 1.4885581155719569 2.0127475000528441 2.5678004898372619 3.1759640344395761 3.8947232743627112 1.3818322190127139e-05 0.00033463977301153642 0.003136065178230593 0.01621509887660184 0.053385470106561229 0.1203807665246763 0.19346465399685869 0.22613897444373929 0.19346465399685869 0.1203807665246763 0.053385470106561229 0.01621509887660184 0.003136065178230593 0.00033463977301153642 1.3818322190127139e-05
"""


L=ss.split('\n')
L = [s for s in L if len(s)>0]

for i in range(len(L)):
    L[i]=[float(g) for g in L[i].split(' ')]
    

for i in range(len(L)):
    D['rule3'][int(L[i][0])]={}
    N = int(L[i][0])
    D['rule3'][int(L[i][0])]['stds'] = [L[i][1]]*N
    D['rule3'][int(L[i][0])]['means'] = L[i][2:2+N]
    D['rule3'][int(L[i][0])]['wts'] = L[i][2+N:2+2*N]
    
with open('GaussianSplitLib1D_ryanruss.pkl','wb') as F:
    pkl.dump(D,F)