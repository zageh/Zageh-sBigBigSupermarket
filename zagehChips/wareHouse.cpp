#include "supermarketSystem.h"

void initialWareHouse(){
    vector<chips> warehouseList;

    chips bestofall;
    bestofall.name="乐事薯片 酸奶油洋葱味 薯片";
    bestofall.quality=5.0;
    bestofall.assessment="夯中之夯，王中之王，我想不出来原味怎么才能赢过他。完完全全的味觉更加丰富的原味，浓浓的奶油香，口感也比原味要厚一点点，虽然贵，但是值得。";
    warehouse.push_back(bestofall);

    chips lays_sour_cream_onion;
    lays_sour_cream_onion.name = "乐事薯片 酸奶油洋葱味 薯片";
    lays_sour_cream_onion.quality = 4.3;
    lays_sour_cream_onion.assessment = "含在嘴里时有类似其他奶油味薯片的香甜感。但是如果不含而是直接咀嚼的话，又会有比较合适浓度的葱香味。吃起来比较咸，像是原味乐事的特殊版。考虑到价格，再好吃也只能给到4.3分。";
    warehouse.push_back(lays_sour_cream_onion);

    chips lays_ketchup;
    lays_ketchup.name = "乐事薯片 亨氏番茄酱味 薯片";
    lays_ketchup.quality = 4.8;
    lays_ketchup.assessment = "我感觉这包薯片味道特别好，5分能打4.8 番茄酱的味道很浓，比鸡汁番茄味的要浓得多，有点像番茄味的好多鱼，但是薯片的口感绝对远超好多鱼。";
    warehouse.push_back(lays_ketchup);

    chips lays_classic;
    lays_classic.name = "乐事薯片 经典原味 薯片";
    lays_classic.quality = 4.9;
    lays_classic.assessment = "直接给到夯！";
    warehouse.push_back(lays_classic);

    chips lays_chicken_tomato;
    lays_chicken_tomato.name = "乐事薯片 鸡汁番茄味 薯片";
    lays_chicken_tomato.quality = 4.0;
    lays_chicken_tomato.assessment = "吃起来挺香的，番茄味不能说很浓，但是非常有特点。总体来说好吃。";
    warehouse.push_back(lays_chicken_tomato);

    chips lays_spicy_hotpot;
    lays_spicy_hotpot.name = "乐事薯片 飘香麻辣锅味 薯片";
    lays_spicy_hotpot.quality = 4.1;
    lays_spicy_hotpot.assessment = "这个薯片的样子非常不像其他的乐事一样平整，但是吃起来非常香，有点麻，很爽。但是一口气吃多了可能会有点腻。我给4.1分。";
    warehouse.push_back(lays_spicy_hotpot);

    chips lays_italian_stew;
    lays_italian_stew.name = "乐事薯片 意大利红烩味 薯片";
    lays_italian_stew.quality = 4.2;
    lays_italian_stew.assessment = "这个味道闻起来是最香的，吃起来没有闻起来香。总体来说还不错，给4.2分。";
    warehouse.push_back(lays_italian_stew);

    chips lays_cucumber;
    lays_cucumber.name = "乐事薯片 黄瓜味 薯片";
    lays_cucumber.quality = 5.0;
    lays_cucumber.assessment = "夯！";
    warehouse.push_back(lays_cucumber);

    chips lays_lime;
    lays_lime.name = "乐事薯片 青柠味 薯片";
    lays_lime.quality = 4.8;
    lays_lime.assessment = "这个味道非常特别，有的时候我特别想吃，有的时候我又特别不想吃。但当我想吃的时候简直无敌了给4.8分。";
    warehouse.push_back(lays_lime);

    chips lays_bbq;
    lays_bbq.name = "乐事薯片 烧烤味 薯片";
    lays_bbq.quality = 3.9;
    lays_bbq.assessment = "一般，所有烧烤味都一般";
    warehouse.push_back(lays_bbq);

    chips lays_spicy_crayfish;
    lays_spicy_crayfish.name = "乐事薯片 香辣小龙虾味 薯片";
    lays_spicy_crayfish.quality = 4.4;
    lays_spicy_crayfish.assessment = "想得很，完完全全烧烤味的上位替代。总体来说还不错，给4.4分。";
    warehouse.push_back(lays_spicy_crayfish);

    chips lays_wave_squid;
    lays_wave_squid.name = "乐事薯片 大波浪鱿鱼味 薯片";
    lays_wave_squid.quality = 3.5;
    lays_wave_squid.assessment = "最拉的乐事，一口气吃多了很长时间都会讨厌这个薯片，完全没有鱿鱼味。但是薯片的形状很有趣，总体来说一般，给3.5分。";
    warehouse.push_back(lays_wave_squid);

    chips angel_spicy;
    angel_spicy.name = "天使薯片 椒香麻辣味 薯片";
    angel_spicy.quality = 3.0;
    angel_spicy.assessment = "不如子弟，还买5块一小包，没有玉昆队徽我永远都不会买。总体来说很一般，给3分。";
    warehouse.push_back(angel_spicy);

    chips angel_original;
    angel_original.name = "天使薯片 原味 薯片";
    angel_original.quality = 3.2;
    angel_original.assessment = "不如子弟，还买5块一小包，没有玉昆队徽我永远都不会买。总体来说很一般，给3.2分。";
    warehouse.push_back(angel_original);

    chips zidi_original;
    zidi_original.name = "子弟薯片 原味 薯片";
    zidi_original.quality = 4.1;
    zidi_original.assessment = "真正意义上的原味，好吃，不像其他牌子的原味都加了很多调料。总体来说还不错，给4分。";
    warehouse.push_back(zidi_original);

    chips zidi_bbq;
    zidi_bbq.name = "子弟薯片 烧烤味 薯片";
    zidi_bbq.quality = 3.8;
    zidi_bbq.assessment = "烧烤味一般般。总体来说一般，给3.8分。";
    warehouse.push_back(zidi_bbq);

    chips zidi_spicy;
    zidi_spicy.name = "子弟薯片 椒香麻辣味 薯片";
    zidi_spicy.quality = 4.2;
    zidi_spicy.assessment = "薯片部分一点也不重要，重要的是最后剩下的渣渣，薯片本身酱料涂的根本不均匀，但是渣渣简直是仙品。";
    warehouse.push_back(zidi_spicy);

    chips zidi_tomato;
    zidi_tomato.name = "子弟薯片 美滋番茄味 薯片";
    zidi_tomato.quality = 3.6;
    zidi_tomato.assessment = "比较一般，番茄味中最弱的一个";
    warehouse.push_back(zidi_tomato);

    chips zidi_blackpepper_beef;
    zidi_blackpepper_beef.name = "子弟薯片 黑椒牛扒味 薯片";
    zidi_blackpepper_beef.quality = 3.9;
    zidi_blackpepper_beef.assessment = "渣渣比较好吃，但是没有麻辣味的好，总体一般";
    warehouse.push_back(zidi_blackpepper_beef);

    chips zidi_cucumber;
    zidi_cucumber.name = "子弟薯片 清怡黄瓜味 薯片";
    zidi_cucumber.quality = 4.2;
    zidi_cucumber.assessment = "很特别的味道，好久没吃了，但是按印象来说值得一尝";
    warehouse.push_back(zidi_cucumber);

    chips lays_brazil_bbq;
    lays_brazil_bbq.name = "乐事薯片 巴西烤肉味 薯片";
    lays_brazil_bbq.quality = 3.4;
    lays_brazil_bbq.assessment = "除了一般我想不出其他形容";
    warehouse.push_back(lays_brazil_bbq);

    chips guiyun_potato;
    guiyun_potato.name = "贵云薯片 贵州特产土豆片";
    guiyun_potato.quality = 4.9;
    guiyun_potato.assessment = "我现在最怀念的下关一中小卖铺零食，升级版的麻辣味子弟。薯片都被炸的有点透明了，酱料也很足，但是吃多了有点咸";
    warehouse.push_back(guiyun_potato);

    chips yunguitechan;
    yunguitechan.name="云贵特产土豆片 找不到名字和图片了 炸的是老鼠洋芋";
    yunguitechan.quality=4.6;
    yunguitechan.assessment="原始的土豆片，老鼠洋芋的样子很独特。炸的不透，味道也很独特，有机会的话值得一尝。但是我不知道下关一中小卖铺是怎么进货进到这样淘宝上都搜不到的小众零食的";

    chips crispy_honey_butter;
    crispy_honey_butter.name = "脆生生薯片 蜂蜜黄油味 薯片";
    crispy_honey_butter.quality = 4.1;
    crispy_honey_butter.assessment = "很香甜，等于北海道乐事减去原味乐事，所有脆生生中最好吃的一包";
    warehouse.push_back(crispy_honey_butter);

    chips crispy_seasalt_lemon;
    crispy_seasalt_lemon.name = "脆生生薯片 海盐柠檬味 薯片";
    crispy_seasalt_lemon.quality = 3.7;
    crispy_seasalt_lemon.assessment = "不是乐事的柠檬味薯片都不是很好吃";
    warehouse.push_back(crispy_seasalt_lemon);

    chips crispy_herb_seasalt;
    crispy_herb_seasalt.name = "脆生生薯片 法香海盐味 薯片";
    crispy_herb_seasalt.quality = 4.0;
    crispy_herb_seasalt.assessment = "比较中规中矩的咸味薯片，算得上好吃，但是不惊艳，同一赛道上竞争对手太多了";
    warehouse.push_back(crispy_herb_seasalt);

    chips lumila_seaweed;
    lumila_seaweed.name = "噜咪啦薯片 海苔味 薯片";
    lumila_seaweed.quality = 4.4;
    lumila_seaweed.assessment = "在小卖铺比较无力的时候是可以称王的，好吃，海苔味很足，就是薯片炸的不够香";
    warehouse.push_back(lumila_seaweed);

    chips lumila_corn;
    lumila_corn.name = "噜咪啦薯片 玉米味 薯片";
    lumila_corn.quality = 3.9;
    lumila_corn.assessment = "好吃，只是我本人没有那么喜欢玉米味的零食而已，绝对有4分以上的实力";
    warehouse.push_back(lumila_corn);

    chips kobik_original;
    kobik_original.name = "可比克薯片 原滋味 薯片";
    kobik_original.quality = 3.9;
    kobik_original.assessment = "就是这个淀粉味爽，爱吃，但是不觉得能算得上好吃，其他可比克也是这样的";
    warehouse.push_back(kobik_original);

    chips shanghaojia_bbq;
    shanghaojia_bbq.name = "上好佳田园薯片 烤肉味 薯片";
    shanghaojia_bbq.quality = 3.9;
    shanghaojia_bbq.assessment = "买空气送薯片。薯片还那么薄，可是酱料真的很足，其他上好佳也是这种类型的";
    warehouse.push_back(shanghaojia_bbq);
}