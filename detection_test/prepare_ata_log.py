from collections import defaultdict
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
import numpy as np
from tools.utils_convert import read_nu_association, read_mu, read_rpi


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Bin
    df_bin = read_rpi(cfg.rpi_all_results_csv, scale=3)
    # df_bin["frame"] = df_bin[
    #     "frame"] + 50  # FIXME: For some reason, there is a 50 frame lag

    # PAX
    df_pax = read_mu(cfg.mu_result_file)

    # NU
    asso_info, theft_info = read_nu_association(cfg.neu_result_file)

    # -------------------------------- Create Log -------------------------------- #
    full_log = defaultdict(list)

    df_comb = pd.concat((df_bin, df_pax),
                        ignore_index=True).sort_values('frame')

    for _, row in tqdm(df_comb.iterrows(),
                       total=df_comb.shape[0],
                       desc="Processing : "):
        camera, frame, _id, x1, y1, x2, y2, _type, _class = row[[
            'camera', 'frame', 'id', 'x1', 'y1', 'x2', 'y2', 'type', 'class'
        ]]
        cam = camera[3:5]  # 'cam09' --> '09'

        if _type not in ("loc", "chng", "empty"):
            continue

        if _class == 'dvi':
            type_log = 'DVI'
        else:
            if _class == 'tso':
                type_log = "TSO"
            else:
                type_log = "PAX"

        if _class == 'dvi':
            pax_id = "NA"

            # owner's id is collected from camera 09
            if _id in asso_info["cam09"]:
                ffs = list(asso_info["cam09"][_id])
                for _f in ffs:
                    if frame >= _f:
                        pax_id = asso_info["cam09"][_id][_f]
            if _type == "loc":
                # LOC: type: DVI camera-num: 11 frame: 3699 time-offset: 123.3 BB: 1785, 258, 1914, 549
                # ID: B2 PAX-ID: P1 left-behind: false
                if camera == "cam13" and frame > 9410 and _id == "B27" and conf.file_num == "exp2":
                    log_msg = (
                        f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                        +
                        f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id} PAX-ID: {pax_id} "
                        + "left-behind: true")
                else:
                    log_msg = (
                        f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                        +
                        f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id} PAX-ID: {pax_id} "
                        + "left-behind: false"
                    )  # FIXME: left-behind calculation
            elif _type in ("chng", "empty"):
                # XFR: type: FROM camera-num: 13 frame: 4765 time-offset: 158.83
                # BB: 1353, 204, 1590, 462 owner-ID: P2 DVI-ID: B5 theft: FALSE

                owner_id = pax_id

                # FIXME: xfr type to in cam 09 and from in other cameras
                xfr_type = 'TO' if cam == '09' else 'FROM'
                if _type == "empty":
                    xfr_type = 'FROM'
                # check potential theft
                _theft = "FALSE"
                if _id in theft_info[camera]:
                    ffs = theft_info[camera][_id]
                    for _f in ffs:
                        if np.abs(frame - _f) < 100:
                            _theft = "TRUE"
                            pax_id = ffs[_f]
                            xfr_type = 'FROM'
                            del ffs[_f]
                            break

                if pax_id != "NA":
                    # get pax BB
                    paxes = df_comb[(df_comb['class'] == 'pax')
                                    & (df_comb['id'] == pax_id)
                                    & (df_comb['camera'] == camera)]
                    _ind = (paxes['frame'] - frame).abs().idxmin()
                    if np.abs(paxes.loc[_ind]['frame'] - frame) < 30:
                        x1, y1, x2, y2 = paxes.loc[_ind][[
                            'x1', 'y1', 'x2', 'y2'
                        ]]
                        # NOTE: decrease frame number in xfr event
                        log_msg = (
                            f"XFR: type: {xfr_type} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                            +
                            f"BB: {x1}, {y1}, {x2}, {y2} owner-ID: {owner_id} DVI-ID: {_id} theft: {_theft}"
                        )  # REVIEW: 'theft'??

        elif _class in ('pax', 'tso'):
            # LOC: type: PAX camera-num: 13 frame: 4358 time-offset: 145.27 BB: 914, 833, 1190, 1079 ID: P1
            if "TSO" in _id: type_log = "TSO"
            log_msg = (
                f"LOC: type: {type_log} camera-num: {cam} frame: {frame} time-offset: {frame/30:.2f} "
                + f"BB: {x1}, {y1}, {x2}, {y2} ID: {_id}")

        full_log[cam].append(log_msg)


if __name__ == "__main__":
    main()