for i in {1..20}
do
	python3.6 hdmi_online_pos.py $i &
done

wait
echo "All done"
