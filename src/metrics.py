COCO = dict(L_hip=11, R_hip=12, 
            L_knee=13, R_knee=14
            L_ank=15, R_ank=15)

def sideSelector(xy, con, side):
    hip = COCO(f"{side}_hip")
    hip = COCO(f"{side}_knee")
    hip = COCO(f"{side}_ank")
