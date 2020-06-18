local=YES
c2d=YES
check=NO

################### localPicker particle picking with local optimization ######
if [ "${local}" == "YES" ]; then

    if [ ! -d  local ]; then
        mkdir local
    fi
    if [ ! -d  local/aligned ]; then
        mkdir local/aligned
    fi
    rm -f  local/aligned/*_local.star

    if [ ! -d  linked ]; then
        mkdir linked
    fi

    ### use mirograph .star file for picking 

    ### select micrographs with lowest 20 defocus. 
    column=`awk -F "#"  '/_rlnMicrographName/ {print $2}' CtfFind/micrographs_defocus_ctf.star`
    list=`awk -v column=$column '/mrc/ { if (NF>2) {print $column}}' CtfFind/micrographs_defocus_ctf.star`  
    for mrcfile in $list; do

        # picking the directory 
        #for mrcfile in `(ls aligned/*.mrc |head -24 )`; do 


        if [ ! -f local/${mrcfile/.mrc/_local.star} ]; then
            echo $mrcfile   ${mrcfile/.mrc/_local.star}

            ## particle size  in pixels
            python -W ignore Self-Supervised/localpicker.py  --mrc_file=${mrcfile} --particle_size=$ptl_pixel --bin_size=$lp_bin_size  --threshold=$lp_threshold --max_sigma=$lp_max_sigma
        fi

        #cp -f $mrcfile linked/

    done

    #if [ "${extract}" == "YES" ]; then 
    echo CtfFind/micrographs_selected_ctf.star > local/coords_suffix_local.star
    #rm -f local/aligned/*_extract.star

    #### manual check 
    if [ "${check}" == "YES" ]; then
        #particle_diameter angstrom
        #pixel=1.2156
        #ptl_size=150
        rm -f local/micrographs_selected.star
        `which relion_manualpick` --i CtfFind/micrographs_defocus_ctf.star --odir local/ --pickname  local --allow_save   --fast_save --selection local/micrographs_selected.star --scale 0.15 --sigma_contrast 3 --black 0 --white 0 --lowpass 10 --angpix $pixel --ctf_scale 1 --particle_diameter $ptl_size 
    fi

    rm -f local/aligned/*_extract.star local/aligned/*.mrcs  local/particles.star
    # extraction box size in pixels
    mpirun -n 12 `which relion_preprocess_mpi` --i CtfFind/micrographs_selected_ctf.star --coord_dir local/ --coord_suffix  _local.star  --part_star local/particles.star  --part_dir  local/ --extract --extract_size  $box_size --norm --bg_radius 25 --white_dust -1 --black_dust -1 --invert_contrast  --scale 64  

fi  
################## end of local particle picking ############

#############################  Class2D by relion  ############################# 
if [ "${c2d}" == "YES" ]; then  
    rm -f log_local.txt
    if [ ! -d  Select ]; then
        mkdir  Select
    fi
    if [ ! -d  Class2D ]; then
        mkdir  Class2D
    fi

    ## Select extracted particles for iterative 2D purification better than 10 angstraom ###
    column=`awk -F "#"  '/_rlnCtfMaxResolution/ {print $2}' local/particles.star`
    #awk -v column=$column '{ if (NF>2 && strtonum($column) == class) print}' Class2D/c2d_search_it025_data.star  >>  tmp.star 
    #column=7
    awk -v column=$column '{ if (NF<=2) {print} else { if (strtonum($column) <= 4) print }}' local/particles.star > Select/particles_selected.star
    #awk -v column=$column '{ if (NF<=2) {print} else { if (strtonum($column) <= 4) print }}' Extract/particles.star > Extract/particles_selected.star
    numTotal=`awk '{ if (NF > 2) print}' Select/particles_selected.star |wc -l`
    numClasses=$(( $numTotal/$ptl_class))
    ## initial 2D class average for statistics
    #module load cuda-7.5 openmpi-1.8.8  relion-2.1.0_gpu

    mpirun  -n 5 `which relion_refine_mpi`  --o Class2D/c2d_local \
                                            --i local/particles.star \
                                            --dont_combine_weights_via_disc \
                                            --no_parallel_disc_io \
                                            --preread_images  --ctf  --fast_subsets  --pool 30 --pad 2 --iter 25  \
                                            --only_flip_phases  \
                                            --ctf_intact_first_peak \
                                            --tau2_fudge 2  \
                                            --fast_subsets \
                                            --particle_diameter $ptl_size \
                                            --K  $numClasses --flatten_solvent  --zero_mask  --oversampling 1 \
                                            --psi_step 12 --offset_range 6 --offset_step 2 --norm --scale \
                                            --j 2 --gpu  "0:1:2:3"

    ### iteration of 2D class average until most particles are in major classes
    while true; do

        list=`fgrep mrcs Class2D/c2d_local_it025_model.star |fgrep -v inf |awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5); if (100*$2/$5 >= 0.1 ) print $1, $2 }' |awk '{split($0,a,"@"); printf "%i\n", a[1]}'`

        numGood=`echo "${list}" |wc -l`
        selectPercent=`fgrep mrcs Class2D/c2d_local_it025_model.star |fgrep -v inf |awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5); if (100*$2/$5 >= 0.1 ) print $1, $2 }'|awk '{sum += $2; n++ } END { if (n > 0) print sum; }'`

        #list=`awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5); if (100*$2/$5 >= 0.1 ) print $1, $2 }' Class2D/c2d_search_it025_model.star |awk '{split($0,a,"@"); printf "%i\n", a[1]}'`
        #selectPercent=`awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5); if (100*$2/$5 >= 0.1 ) print $1, $2 }' Class2D/c2d_search_it025_model.star |awk '{sum += $2; n++ } END { if (n > 0) print sum; }'`

        awk '{ if (NF<=2) print}' Class2D/c2d_local_it025_data.star > Select/particles_selected.star
        column=`awk -F "#"  '/_rlnClassNumber/ {print $2}' Select/particles_selected.star`
        for class in `(echo "$list")`; do
            #echo $class
            awk -v class=$class -v column=$column '{ if (NF>2 && strtonum($column) == class) print}' Class2D/c2d_local_it025_data.star  >> Select/particles_selected.star
        done

        #python /share/apps/autoEM/star2particle.py --ref  /share/d2/cryoarm200/relion3/relion30_tutorial/shiny_rename.star \
        #  --dist 20  --comp  Select/particles_selected.star

        if [ "${check}" == "YES" ]; then
            ### manual select classes 
            `which relion_display` --gui --i Class2D/c2d_local_it025_model.star --allow_save --fn_parts  Select/particles_selected.star
        fi

        #numTotal=`awk '{ if (NF > 2) print}' Select/particles_selected.star |wc -l`
        numParticles=`awk '{ if (NF > 2) print}'  Select/particles_selected.star |wc -l`
        #numClasses=$(( $numGood * 2))
        numClasses=$(( $numParticles/$ptl_class)); echo $numClasses
        echo  "number of particles:" ${numParticles} " "  ${selectPercent} "percentage of: " ${numTotal}   >> log_local.txt
        #fgrep mrcs Class2D/c2d_local_it025_model.star |awk '{gsub(/inf/,"100")}1' |awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5);  print $1, $2, $5,  100*$2/$5 }' |sort -k4 -r >> log_local.txt
        fgrep mrcs Class2D/c2d_local_it025_model.star |fgrep -v inf |awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5);  print $1, $2, $5,  100*$2/$5 }' |sort -k4 -r >> log_local.txt
        numTotal=`awk '{ if (NF > 2) print}' Select/particles_selected.star|wc -l`
        if (( $(bc <<< "${selectPercent} < 0.90") )); then 

            ### re-extract selected particles

            if [ ! -d  Extract ]; then
                mkdir Extract
            fi
            if [ ! -d  Extract/aligned ]; then
                mkdir Extract/aligned
            fi
            rm -f Extract/aligned/*_extract.star Extract/aligned/*.mrcs  Extract/particles.star

            mpirun -n 5 `which relion_preprocess_mpi` --i CtfFind/micrographs_selected_ctf.star --reextract_data_star Select/particles_selected.star   --part_star Extract/particles.star --part_dir Extract/  --extract --extract_size $box_size --scale 64 --norm --bg_radius 25  --white_dust -1 --black_dust -1 --invert_contrast   --recenter --recenter_x 0 --recenter_y 0 --recenter_z 0 

            ### percentage of good particles:
            #python /share/apps/autoEM/star2particle.py --ref  /share/d2/cryoarm200/relion3/relion30_tutorial/shiny_rename.star  --dist 20  --#comp Extract/particles.star |tee tmp.log

            mpirun  -n 5 `which relion_refine_mpi`  --o Class2D/c2d_local \
                                                    --i Extract/particles.star  \
                                                    --dont_combine_weights_via_disc \
                                                    --no_parallel_disc_io \
                                                    --preread_images  --ctf  --fast_subsets   --pool 30 --pad 2 --iter 25  \
                                                    --only_flip_phases  \
                                                    --ctf_intact_first_peak \
                                                    --tau2_fudge 2  \
                                                    --fast_subsets \
                                                    --particle_diameter $ptl_size \
                                                    --K $numClasses --flatten_solvent  --zero_mask  --oversampling 1 \
                                                    --psi_step 12 --offset_range 6 --offset_step 2 --norm --scale \
                                                    --j 2 --gpu  "0:1:2:3"
            #  --strict_highres_exp 10
            echo "here"
            numTotal=`awk '{ if (NF > 2) print}' Class2D/c2d_local_it025_data.star |wc -l`
        else

            cp -f Class2D/c2d_local_it025_model.star Class2D/c2d_search_it025_model.star
            cp -f Class2D/c2d_local_it025_data.star Class2D/c2d_search_it025_data.star
            cp -f Select/particles_selected.star  particles_selected.star

            if [ "${check}" == "YES" ]; then
                `which relion_display` --gui --i Class2D/c2d_local_it025_model.star --allow_save --fn_parts Select/particles_selected.star --fn_imgs Select/class_averages.star --recenter  --regroup  5
            fi

            break
        fi
    done ## finish particle purification

fi ## END OF C2D  

