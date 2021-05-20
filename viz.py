import numpy as np
import seaborn as sns

from numpy import linalg
from lsdo_viz.api import BaseViz, Frame

sns.set()


class Viz(BaseViz):
    def setup(self):

        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=8.,
                width_in=12.,
                nrows=3,
                ncols=3,
                wspace=0.4,
                hspace=0.4,
            ), 1)

    def plot(self,
             data_dict_current,
             data_dict_all,
             limits_dict,
             ind,
             video=False):
                t = data_dict_current['integrator_group.times']
                rx = data_dict_current['integrator_group.state:rx']
                ry = data_dict_current['integrator_group.state:ry']
                rz = data_dict_current['integrator_group.state:rz']
                vx = data_dict_current['integrator_group.state:Vx']
                vy = data_dict_current['integrator_group.state:Vy']
                vz = data_dict_current['integrator_group.state:Vz']
                m = data_dict_current['integrator_group.state:m']
                ux = data_dict_current['ux_comp.ux']
                uy = data_dict_current['uy_comp.uy']
                uz = data_dict_current['uz_comp.uz']
                C1 = data_dict_current['ConstraintOne.C1']
                C2 = data_dict_current['ConstraintTwo.C2']
                C2 = data_dict_current['ConstraintThree.C3']
                CU = data_dict_current['ConstraintUnit.CU']

                tscale = 1034.214677907832
                rscale = 1738100.
                vscale = 1680.598851600226
                mscale = 15103.
                a = 1.04603
                ecc = 0.0385

                self.get_frame(1).clear_all_axes()

# First Row

                with self.get_frame(1)[0, 0] as ax:
                    r = np.sqrt(rx[:,0]**2+ry[:,0]**2+rz[:,0]**2)
                    sns.lineplot(x=tscale*t, y=rscale*(r - 1.), ax=ax)
                    if video:
                        limits = self.get_limits(
                            ['integrator_group.times'],
                            fig_axis=0,
                            data_axis=0,
                            mode='broad', # final
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * tscale
                            for ind in range(2)
                        ]
                        ax.set_xlim(new_limits)

                        # ax.set_ylim(
                        #     self.get_limits(
                        #         ['integrator_group.state:rx'],
                        #         fig_axis=0,
                        #         data_axis=0,
                        #         mode='broad',
                        #         lower_margin=0.1,
                        #         upper_margin=0.1,
                        #     ), )

                        ax.set_xlabel('t (s)')
                        ax.set_ylabel('r norm (m)')

                with self.get_frame(1)[0, 1] as ax:
                    v = np.sqrt(vx[:,0]**2+vy[:,0]**2+vz[:,0]**2)
                    sns.lineplot(x=tscale*t, y=vscale*v, ax=ax)
                    if video:
                        limits = self.get_limits(
                            ['integrator_group.times'],
                            fig_axis=0,
                            data_axis=0,
                            mode='broad',
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * tscale
                            for ind in range(2)
                        ]
                        ax.set_xlim(new_limits)

                        # limits = self.get_limits(
                        #     ['integrator_group.state:Vx'],
                        #     fig_axis=0,
                        #     data_axis=0,
                        #     mode='broad',
                        #     lower_margin=0.1,
                        #     upper_margin=0.1,
                        # )
                        # new_limits = [
                        #     limits[ind] * vscale
                        #     for ind in range(2)
                        # ]
                        # ax.set_ylim(new_limits)

                        ax.set_xlabel('t (s)')
                        ax.set_ylabel('v norm (m/s)')

                with self.get_frame(1)[0, 2] as ax:
                    sns.lineplot(x=tscale*t, y=mscale*m[:,0], ax=ax)
                    if video:
                        limits = self.get_limits(
                            ['integrator_group.times'],
                            fig_axis=0,
                            data_axis=0,
                            mode='broad',
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * tscale
                            for ind in range(2)
                        ]
                        ax.set_xlim(new_limits)

                        limits = self.get_limits(
                            ['integrator_group.state:m'],
                            fig_axis=0,
                            data_axis=0,
                            mode='broad',
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * mscale
                            for ind in range(2)
                        ]
                        ax.set_ylim(new_limits)

                        ax.set_xlabel('t (s)')
                        ax.set_ylabel('mass (kg)')

# Second Row

                with self.get_frame(1)[1, 0] as ax:
                    sns.lineplot(x=tscale*t, y=ux[:,0], ax=ax)
                    if video:
                        limits = self.get_limits(
                            ['integrator_group.times'],
                            fig_axis=0,
                            data_axis=0,
                            mode='broad',
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * tscale
                            for ind in range(2)
                        ]
                        ax.set_xlim(new_limits)
                        ax.set_ylim(
                            self.get_limits(
                                ['ux_comp.ux'],
                                fig_axis=0,
                                data_axis=0,
                                mode='broad',
                                lower_margin=0.1,
                                upper_margin=0.1,
                            ), )
                    ax.set_xlabel('t')
                    ax.set_ylabel('ux')

                with self.get_frame(1)[1, 1] as ax:
                    sns.lineplot(x=tscale*t, y=uy[:,0], ax=ax)
                    if video:
                        limits = self.get_limits(
                            ['integrator_group.times'],
                            fig_axis=0,
                            data_axis=0,
                            # mode='broad',
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * tscale
                            for ind in range(2)
                        ]
                        ax.set_xlim(new_limits)
                        ax.set_ylim(
                            self.get_limits(
                                ['uy_comp.uy'],
                                fig_axis=0,
                                data_axis=0,
                                mode='broad',
                                lower_margin=0.1,
                                upper_margin=0.1,
                            ), )

                    ax.set_xlabel('t')
                    ax.set_ylabel('uy')

                with self.get_frame(1)[1, 2] as ax:
                    sns.lineplot(x=tscale*t, y=uz[:,0], ax=ax)
                    if video:
                        limits = self.get_limits(
                            ['integrator_group.times'],
                            fig_axis=0,
                            data_axis=0,
                            mode='broad',
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * tscale
                            for ind in range(2)
                        ]
                        ax.set_xlim(new_limits)
                        ax.set_ylim(
                            self.get_limits(
                                ['uz_comp.uz'],
                                fig_axis=0,
                                data_axis=0,
                                mode='broad',
                                lower_margin=0.1,
                                upper_margin=0.1,
                            ), )

                    ax.set_xlabel('t')
                    ax.set_ylabel('uz')

# Third Row

                with self.get_frame(1)[2, 0] as ax:
                    sns.lineplot(x=tscale*t, y=(ux[:,0]**2 + uy[:,0]**2 + uz[:,0]**2), ax=ax)
                    if video:
                        limits = self.get_limits(
                            ['integrator_group.times'],
                            fig_axis=0,
                            data_axis=0,
                            # mode='broad',
                            lower_margin=0.1,
                            upper_margin=0.1,
                        )
                        new_limits = [
                            limits[ind] * tscale
                            for ind in range(2)
                        ]
                        ax.set_xlim(new_limits)

                        # limits = self.get_limits(
                        #     ['ux_comp.ux'],
                        #     fig_axis=0,
                        #     data_axis=0,
                        #     mode='broad',
                        #     lower_margin=0.1,
                        #     upper_margin=0.1,
                        # )
                        # new_limits = [
                        #     2. * np.abs(limits[ind])
                        #     for ind in range(2)
                        # ]
                        # ax.set_xlim(new_limits)

                        # ax.set_ylim(
                        #     self.get_limits(
                        #         ['ux_comp.ux'],
                        #         fig_axis=0,
                        #         data_axis=0,
                        #         lower_margin=0.1,
                        #         upper_margin=0.1,
                        #         mode='broad',
                        #     ), )
                    ax.set_xlabel('t (s)')
                    ax.set_ylabel('u norm')


                # with self.get_frame(1)[2, 1] as ax:
                #     r = [rx, ry, rz]
                #     V = [vx, vy, vz]
                #     rxv = np.cross(r,v)
                #     C1 = np.dot(rxv,rxv) - a*(1 - ecc**2)
                #     sns.lineplot(x=tscale*t, y=C1, ax=ax)
                #     if video:
                #         limits = self.get_limits(
                #             ['integrator_group.times'],
                #             fig_axis=0,
                #             data_axis=0,
                #             mode='final',
                #             lower_margin=0.1,
                #             upper_margin=0.1,
                #         )
                #         new_limits = [
                #             limits[ind] * tscale
                #             for ind in range(2)
                #         ]
                #         ax.set_xlim(new_limits)
                #
                #         # ax.set_ylim(
                #         #     self.get_limits(
                #         #         ['ux_comp.ux'],
                #         #         fig_axis=0,
                #         #         data_axis=0,
                #         #         lower_margin=0.1,
                #         #         upper_margin=0.1,
                #         #         mode='broad',
                #         #     ), )
                #
                #     ax.set_xlabel('t')
                #     ax.set_ylabel('Orbital Condition 1')

                # with self.get_frame(1)[2, 2] as ax:
                #     r = np.sqrt(rx[:,0]**2+ry[:,0]**2+rz[:,0]**2)
                #     v = np.sqrt(vx[:,0]**2+vy[:,0]**2+vz[:,0]**2)
                #     C2 = linalg.norm(v,2)**2/2. - 1./linalg.norm(r,2) + 1./(2*a)
                #     sns.lineplot(x=tscale*t, y=C2, ax=ax)
                #     if video:
                #         limits = self.get_limits(
                #             ['integrator_group.times'],
                #             fig_axis=0,
                #             data_axis=0,
                #             mode='final',
                #             lower_margin=0.1,
                #             upper_margin=0.1,
                #         )
                #         new_limits = [
                #             limits[ind] * tscale
                #             for ind in range(2)
                #         ]
                #         ax.set_xlim(new_limits)
                #
                #         # ax.set_ylim(
                #         #     self.get_limits(
                #         #         ['ux_comp.ux'],
                #         #         fig_axis=0,
                #         #         data_axis=0,
                #         #         lower_margin=0.1,
                #         #         upper_margin=0.1,
                #         #         mode='broad',
                #         #     ), )
                #
                #     ax.set_xlabel('t (s)')
                #     ax.set_ylabel('Orbital Condition 2')

# Fourth Row

                # with self.get_frame(1)[3, 0] as ax:
                #     sns.lineplot(x=tscale*t, y=C1, ax=ax)
                #     if video:
                #         limits = self.get_limits(
                #             ['integrator_group.times'],
                #             fig_axis=0,
                #             data_axis=0,
                #             mode='final',
                #             lower_margin=0.1,
                #             upper_margin=0.1,
                #         )
                #         new_limits = [
                #             limits[ind] * tscale
                #             for ind in range(2)
                #         ]
                #         ax.set_xlim(new_limits)
                #         ax.set_ylim(
                #             self.get_limits(
                #                 ['ConstraintOne.C1'],
                #                 fig_axis=0,
                #                 data_axis=0,
                #                 lower_margin=0.1,
                #                 upper_margin=0.1,
                #                 mode='broad',
                #             ), )
                #     ax.set_xlabel('t')
                #     ax.set_ylabel('C1')
                #
                # with self.get_frame(1)[3, 1] as ax:
                #     sns.lineplot(x=tscale*t, y=C2, ax=ax)
                #     if video:
                #         limits = self.get_limits(
                #             ['integrator_group.times'],
                #             fig_axis=0,
                #             data_axis=0,
                #             mode='final',
                #             lower_margin=0.1,
                #             upper_margin=0.1,
                #         )
                #         new_limits = [
                #             limits[ind] * tscale
                #             for ind in range(2)
                #         ]
                #         ax.set_xlim(new_limits)
                #         ax.set_ylim(
                #             self.get_limits(
                #                 ['ConstraintTwo.C2'],
                #                 fig_axis=0,
                #                 data_axis=0,
                #                 lower_margin=0.1,
                #                 upper_margin=0.1,
                #                 mode='broad',
                #             ), )
                #     ax.set_xlabel('t')
                #     ax.set_ylabel('C2')
                #
                # with self.get_frame(1)[3, 2] as ax:
                #     sns.lineplot(x=tscale*t, y=C3, ax=ax)
                #     if video:
                #         limits = self.get_limits(
                #             ['integrator_group.times'],
                #             fig_axis=0,
                #             data_axis=0,
                #             mode='final',
                #             lower_margin=0.1,
                #             upper_margin=0.1,
                #         )
                #         new_limits = [
                #             limits[ind] * tscale
                #             for ind in range(2)
                #         ]
                #         ax.set_xlim(new_limits)
                #         ax.set_ylim(
                #             self.get_limits(
                #                 ['ConstraintThree.C3'],
                #                 fig_axis=0,
                #                 data_axis=0,
                #                 lower_margin=0.1,
                #                 upper_margin=0.1,
                #                 mode='broad',
                #             ), )
                #     ax.set_xlabel('t')
                #     ax.set_ylabel('C3')

                # Write frames
                self.get_frame(1).write()
 # ---------------------------------------------------------------------------

        # with self.get_frame(1)[0, 0] as ax:
        #     sns.lineplot(x=1034.2*t, y=ux[:,0], ax=ax)
        #     if video:
        #         limits = self.get_limits(
        #             ['integrator_group.times'],
        #             fig_axis=0,
        #             data_axis=0,
        #             mode='final',
        #             lower_margin=0.1,
        #             upper_margin=0.1,
        #         )
        #         new_limits = [
        #             limits[ind] * 1034.2
        #             for ind in range(2)
        #         ]
        #         ax.set_xlim(new_limits)
        #         ax.set_ylim(
        #             self.get_limits(
        #                 ['ux_comp.ux'],
        #                 fig_axis=0,
        #                 data_axis=0,
        #                 lower_margin=0.1,
        #                 upper_margin=0.1,
        #                 mode='broad',
        #             ), )
        #     ax.set_xlabel('t')
        #     ax.set_ylabel('ux')
        #
        # with self.get_frame(1)[1, 0] as ax:
        #     sns.lineplot(x=1034.2*t, y=uy[:,0], ax=ax)
        #     if video:
        #         limits = self.get_limits(
        #             ['integrator_group.times'],
        #             fig_axis=0,
        #             data_axis=0,
        #             mode='final',
        #             lower_margin=0.1,
        #             upper_margin=0.1,
        #         )
        #         new_limits = [
        #             limits[ind] * 1034.2
        #             for ind in range(2)
        #         ]
        #         ax.set_xlim(new_limits)
        #         ax.set_ylim(
        #             self.get_limits(
        #                 ['uy_comp.uy'],
        #                 fig_axis=0,
        #                 data_axis=0,
        #                 mode='broad',
        #                 lower_margin=0.1,
        #                 upper_margin=0.1,
        #             ), )
        #     ax.set_xlabel('t')
        #     ax.set_ylabel('uy')
        #
        # with self.get_frame(1)[2, 0] as ax:
        #     sns.lineplot(x=1034.2*t, y=uz[:,0], ax=ax)
        #     if video:
        #         limits = self.get_limits(
        #             ['integrator_group.times'],
        #             fig_axis=0,
        #             data_axis=0,
        #             mode='final',
        #             lower_margin=0.1,
        #             upper_margin=0.1,
        #         )
        #         new_limits = [
        #             limits[ind] * 1034.2
        #             for ind in range(2)
        #         ]
        #         ax.set_xlim(new_limits)
        #         ax.set_ylim(
        #             self.get_limits(
        #                 ['uz_comp.uz'],
        #                 fig_axis=0,
        #                 data_axis=0,
        #                 mode='broad',
        #                 lower_margin=0.1,
        #                 upper_margin=0.1,
        #             ), )
        #     ax.set_xlabel('t')
        #     ax.set_ylabel('uz')
        #
        # with self.get_frame(1)[0, 2] as ax:
        #     sns.lineplot(x=1034.2*t, y=(ux[:,0]**2 + uy[:,0]**2 + uz[:,0]**2), ax=ax)
        #     if video:
        #         limits = self.get_limits(
        #             ['integrator_group.times'],
        #             fig_axis=0,
        #             data_axis=0,
        #             mode='final',
        #             lower_margin=0.1,
        #             upper_margin=0.1,
        #         )
        #         new_limits = [
        #             limits[ind] * 1034.2
        #             for ind in range(2)
        #         ]
        #         ax.set_xlim(new_limits)
        #         ax.set_ylim(
        #             self.get_limits(
        #                 ['ux_comp.ux'],
        #                 fig_axis=0,
        #                 data_axis=0,
        #                 lower_margin=0.1,
        #                 upper_margin=0.1,
        #                 mode='broad',
        #             ), )
        #     ax.set_xlabel('t')
        #     ax.set_ylabel('u norm')
        #
        # with self.get_frame(1)[1:, 1:] as ax:
        #     r = np.sqrt(rx[:,0]**2+ry[:,0]**2+rz[:,0]**2)
        #     sns.lineplot(x=1034.2*t, y=1738100.*(r - 1.), ax=ax)
        #     if video:
        #         limits = self.get_limits(
        #             ['integrator_group.times'],
        #             fig_axis=0,
        #             data_axis=0,
        #             mode='final',
        #             lower_margin=0.1,
        #             upper_margin=0.1,
        #         )
        #         new_limits = [
        #             limits[ind] * 1034.2
        #             for ind in range(2)
        #         ]
        #         ax.set_xlim(new_limits)
        #
        #         # ax.set_ylim(
        #         #     self.get_limits(
        #         #         ['integrator_group.state:rx'],
        #         #         fig_axis=0,
        #         #         data_axis=0,
        #         #         mode='broad',
        #         #         lower_margin=0.1,
        #         #         upper_margin=0.1,
        #         #     ), )
        #     ax.set_xlabel('t')
        #     ax.set_ylabel('r norm')
        #
        # with self.get_frame(1)[0, 1] as ax:
        #     sns.lineplot(x=1034.2*t, y=15103.*m[:,0], ax=ax)
        #     if video:
        #         limits = self.get_limits(
        #             ['integrator_group.times'],
        #             fig_axis=0,
        #             data_axis=0,
        #             mode='final',
        #             lower_margin=0.1,
        #             upper_margin=0.1,
        #         )
        #         new_limits = [
        #             limits[ind] * 1034.2
        #             for ind in range(2)
        #         ]
        #         ax.set_xlim(new_limits)
        #
        #         # ax.set_ylim(
        #         #     self.get_limits(
        #         #         ['integrator_group.state:vx'],
        #         #         fig_axis=0,
        #         #         data_axis=0,
        #         #         mode='broad',
        #         #         lower_margin=0.1,
        #         #         upper_margin=0.1,
        #         #     ), )
        #     ax.set_xlabel('t')
        #     ax.set_ylabel('mass')
        #
        # # with self.get_frame(1)[0, 2] as ax:
        # #     sns.lineplot(x=1034.2*t, y=aT[:,0], ax=ax)
        # #     if video:
        # #         limits = self.get_limits(
        # #             ['integrator_group.times'],
        # #             fig_axis=0,
        # #             data_axis=0,
        # #             mode='final',
        # #             lower_margin=0.1,
        # #             upper_margin=0.1,
        # #         )
        # #         new_limits = [
        # #             limits[ind] * 1034.2
        # #             for ind in range(2)
        # #         ]
        # #         ax.set_xlim(new_limits)
        # #         ax.set_ylim(
        # #             self.get_limits(
        # #                 ['aT_comp.aT'],
        # #                 fig_axis=0,
        # #                 data_axis=0,
        # #                 lower_margin=0.1,
        # #                 upper_margin=0.1,
        # #                 mode='broad',
        # #             ), )
        # #     ax.set_xlabel('t')
        # #     ax.set_ylabel('thrust acceleration')
