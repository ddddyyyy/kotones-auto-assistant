"""完整文字裁剪的纯图像表示；不解释数值或裁掉字符。"""
import cv2
from cv2.typing import MatLike


def neutral_white_glyphs(pixels: MatLike) -> MatLike:
    return ((pixels.min(axis=2)>245)&(pixels.max(axis=2)-pixels.min(axis=2)<12)).astype('uint8')*255


def right_edge_digit(mask: MatLike) -> bool:
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    height,width=mask.shape
    # 大片亮背景不是文字；仅把贴右边界的数字尺寸连通域视为裁切风险。
    return any(x+w>=width-1 and height*.4<=h<height*.95 and 1<=w<=height*.8 and area>=4
               for x,y,w,h,area in stats[1:])
