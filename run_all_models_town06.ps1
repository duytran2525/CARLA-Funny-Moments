# Model 4: cnn_steeringV2 TL=0.0353 VL=0.0092
python run_agents.py --agent lane_follow --sync --map Town06 --spawn-point 37 --record-video --video-output-path "outputs/town06_model4_V2_TL0.0353_VL0.0092.mp4" --video-fps 30 --video-duration-sec 240 --video-codec XVID --model-path "models/cnn_steeringV2 TL=0.0353 VL=0.0092.pth" --spectator-reapply-each-tick --target-speed-kmh 30 --max-throttle 0.5

# Model 5: cnn_steeringV2 TL=0.0384 VL=0.0092
python run_agents.py --agent lane_follow --sync --map Town06 --spawn-point 37 --record-video --video-output-path "outputs/town06_model5_V2_TL0.0384_VL0.0092.mp4" --video-fps 30 --video-duration-sec 240 --video-codec XVID --model-path "models/cnn_steeringV2 TL=0.0384 VL=0.0092.pth" --spectator-reapply-each-tick --target-speed-kmh 30 --max-throttle 0.5

# Model 6: cnn_steeringV2 TL=0.0392 VL=0.0089
python run_agents.py --agent lane_follow --sync --map Town06 --spawn-point 37 --record-video --video-output-path "outputs/town06_model6_V2_TL0.0392_VL0.0089.mp4" --video-fps 30 --video-duration-sec 240 --video-codec XVID --model-path "models/cnn_steeringV2 TL=0.0392 VL=0.0089.pth" --spectator-reapply-each-tick --target-speed-kmh 30 --max-throttle 0.5
