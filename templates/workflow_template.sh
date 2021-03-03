kpicker=YES
check=NO

########################## ITERATIVE PURIFICATION OF PARTICLES by using CNN & relion ############################

###########################################################################
###### Iterative ktraining, kpicking and 2D class average #################
###########################################################################
if [ "${kpicker}" == "YES" ]; then 

        rm -f log_kpicker.txt 
        rm -f test_model.h5 
        ### Iterative training and particle picking ################
        while true; do
                ### Re-center and Re-extract particles for training
                rm -rf Extract/*
                mpirun -n 12 `which relion_preprocess_mpi` --i CtfFind/micrographs_defocus_ctf.star    --reextract_data_star particles_selected.star  --part_star particles_good.star    --part_dir Extract/ --extract --extract_size $box_size --scale 64 --norm --bg_radius 25    --white_dust -1 --black_dust -1 --invert_contrast   --recenter --recenter_x 0 --recenter_y 0 --recenter_z 0 
                                                        
                # ktraining either from scrach or tuning pre-trained model
                python  Self-Supervised/ktraining_aug.py --train_good 'particles_selected.star' --particle_size $box_size   --logdir  'Logfile'   --bin_size $bin_size --model_save_file 'test_model.h5'
                sleep 1 

                ## create directory for Kpicker
                if [ ! -d  Kpicker ]; then
                        mkdir Kpicker
                fi
                if [ ! -d  Kpicker/aligned ]; then
                        mkdir Kpicker/aligned
                fi

                ## particle picking 
                rm -f Kpicker/aligned/*
                sleep 1 
                python Self-Supervised/kpicking_cpu.py  --input_dir 'linked'  --output_dir 'Kpicker/aligned'  --pre_trained_model 'test_model.h5' --star_file  'CtfFind/micrographs_defocus_ctf.star' --threshold $picking_threshold  --threads $picking_threads  --particle_size  $box_size --coordinate_suffix '_kpicker'   --bin_size $bin_size    # --testRun

                ### Extract Kpicker particles and record the number of particles
                rm -f Kpicker/aligned/*.mrcs  Kpicker/aligned/*extract.star  Kpicker/particles.star
                mpirun -n 12 `which relion_preprocess_mpi` --i CtfFind/micrographs_defocus_ctf.star   --coord_dir Kpicker/  --coord_suffix  _kpicker.star  --part_star  Kpicker/particles.star  --part_dir Kpicker/ --extract  --extract_size  $box_size --norm --bg_radius 25  --white_dust -1 --black_dust -1 --invert_contrast  --scale 64 

                numberKpicker=`awk '{ if (NF > 2) print}'  Kpicker/particles.star |wc -l`
                ##  Iterative 2D class averages to clean up training templates

                if [ "${check}" == "YES" ]; then
                        ## Display Kpicker extracted particles
                        rm -f Kpicker/micrographs_selected.star
                        `which relion_manualpick` --i CtfFind/micrographs_defocus_ctf.star --odir Kpicker/ --pickname kpicker --allow_save   --fast_save --selection Kpicker/micrographs_selected.star --scale 0.15 --sigma_contrast 3 --black 0 --white 0 --lowpass 10 --angpix  $pixel --ctf_scale 1 --particle_diameter $ptl_size
                fi

                while true; do

                        ### validation
                        #python /share/apps/autoEM/star2particle.py --ref  /share/d2/cryoarm200/relion3/relion30_tutorial/shiny_rename.star  --dist 20  --comp Kpicker/particles.star >> log_kpicker.txt

                        #echo ${ctf_file} > Kpicker/coords_suffix_autopick.star

                        ## ## Select 1500 extracted particles for iterative 2D purification ###
                        #awk '{  if (NF<=2) print}' Kpicker/particles.star > Kpicker/particles_selected.star
                        #awk '{  if (NF > 2) print}' Kpicker/particles.star  >> Kpicker/particles_selected.star
                        #echo "selected 5000 particles for 2D class averaging ...." 
                        #sleep 1

                        if [ ! -d Class2D ]; then
                                mkdir Class2D
                        fi

                        ### Calculate the number of classes and do 2D averaging. 
                        numParticles=`awk '{ if (NF > 2) print}'  Kpicker/particles.star |wc -l`
                        numClasses=$(( $numParticles/$ptl_class))
                        rm -f Class2D/c2d_search*
                        mpirun  -n 5 `which relion_refine_mpi`  --o Class2D/c2d_search  --i Kpicker/particles.star   --dont_combine_weights_via_disc  --no_parallel_disc_io  --preread_images  --ctf  --pool 30 --pad 2 --iter 25   --only_flip_phases   --ctf_intact_first_peak  --tau2_fudge 2   --fast_subsets  --particle_diameter $ptl_size  --K $numClasses  --flatten_solvent  --zero_mask  --oversampling 1  --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 2 --gpu  "0:1:2:3"  --strict_highres_exp 10
                        # --strict_highres_exp 10

                        ### Select particles based on class distributions ################
                        list=`fgrep mrcs Class2D/c2d_search_it025_model.star |fgrep -v inf |awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5); if (100*$2/$5 >= 0.1 ) print $1, $2 }' |awk '{split($0,a,"@"); printf "%i\n", a[1]}'`
                        selectPercent=`fgrep mrcs Class2D/c2d_search_it025_model.star |fgrep -v inf |awk '/mrcs/ {$2=strtonum($2); $5=strtonum($5); if (100*$2/$5 >= 0.1 ) print $1, $2 }'|awk '{sum += $2; n++ } END { if (n > 0) print sum; }'`
                        ### get good particles
                        awk '{ if (NF<=2) print}' Class2D/c2d_search_it025_data.star > particles_selected.star
                        column=`awk -F "#"  '/_rlnClassNumber/ {print $2}' particles_selected.star`
                        for class in `(echo "$list")`; do
                                awk -v class=$class -v column=$column '{ if (NF>2 && strtonum($column) == class) print}' Class2D/c2d_search_it025_data.star  >>  particles_selected.star 
                        done

                        ### manual selection of 2D classes to override automatically selected. 
                        if [ "${check}" == "YES" ]; then
                                `which relion_display` --gui --i Class2D/c2d_search_it025_model.star --allow_save  --fn_parts  particles_selected.star  --fn_imgs Select/class_averages.star --recenter  --regroup 5
                        fi 

                        numParticles=`awk '{ if (NF > 2) print}'  particles_selected.star |wc -l`
                        numTotal=`awk '{ if (NF > 2) print}' Class2D/c2d_search_it025_data.star |wc -l`
                        numClasses=$(( $numParticles/$ptl_class)); echo $numClasses
                        echo  "number of particles after 2D polish: " ${numParticles} " "  ${selectPercent} "percentage of: " ${numTotal} >> log_kpicker.txt

                        if (( $(bc <<< "${selectPercent} > 0.90") )); then
                                break
                        else 
                                ## Re-extraction of picked particles for 2D if not good enough
               rm -f Kpicker/particles.star Kpicker/aligned/*.mrcs   Kpicker/aligned/*extract.star  
                                mpirun -n 12 `which relion_preprocess_mpi` --i CtfFind/micrographs_defocus_ctf.star   --reextract_data_star particles_selected.star  --part_star Kpicker/particles.star --part_dir Kpicker/   --extract --extract_size $box_size --scale 64 --norm --bg_radius 25  --white_dust -1 --black_dust -1 --invert_contrast   --recenter --recenter_x 0 --recenter_y 0 --recenter_z 0
                        fi  
                done ### END of Class2D purification


                ### Continue to training and picking the total percentage is satisified. 
                picker_acc=`echo "scale=2; $numParticles / $numberKpicker " |bc`

                echo  "number of good particles from kpicker: " ${numParticles} " "  ${picker_acc} "percentage of: " ${numberKpicker} >> log_kpicker.txt



                if (( $(bc <<< "${picker_acc} > 0.7 ") )); then 

                        if [ ! -d Select ]; then
                                mkdir Select
                        fi 

                        if [ "${check}" == "YES" ]; then
                                ## display and final selection of template 
                                `which relion_display` --gui --i Class2D/c2d_search_it025_model.star --allow_save --fn_parts  particles_selected.star --fn_imgs Select/class_averages.star --recenter  --regroup 5
                        fi 

                        #exit ### for test

                        break
                fi
        done ## finish particle purification
fi ## END OF iterative classification and kpicking 
