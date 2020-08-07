#!/bin/bash

IS=128

TEXT="$(echo "The forms of printed letters should be beautiful, and \
that their arrangement \
on the page should be reasonable and a help to the \
shapeliness of the letters \
themselves. The forms of printed letters should be \
beautiful, and that their \
arrangement on the page should be reasonable and a \
help to the shapeliness of \
the letters themselves." | head -c ${IS})"

INPUT="benchmark.txt"

if [[ "${#}" != 1 ]]; then
  echo "Invalid number of arguments '${#}'" 
  echo "Usage:"
  echo "\t$0 <batch size>"
  exit 1
fi

BS="${1}"

if [[ -f "${INPUT}" ]]; then
  rm ${INPUT}
fi

for i in $(seq ${BS}); do
  echo "${TEXT}." >> "${INPUT}" 
done

(
for i in {1..1000}; do
  ./run_trtis_client.sh "${INPUT}" "${BS}"
done
) | awk 'BEGIN{i=0}
         /Total Processing time:/{ t[$4 "_" i] = $4; total+=$4;}
         /Processed [0-9]* sequences for a total of/{ v=($8/$2); len+=v; ++i; } 
         END {
            i = 0;
            asort(t);
            for (k in t) {
              times[i] = t[k];
              ++i;
            }
            print("Performed " i " runs.");
            as = len/i;
            avg = total/i;
            l50 = times[int(i*0.50)];
            l90 = times[int(i*0.90)];
            l95 = times[int(i*0.95)];
            l99 = times[int(i*0.99)];
            mels = int(as * 86.6);
            rtf = as / avg;

            std =0;
            for (k in times) {
              v = times[k];
              std += (v-avg)*(v-avg)
            }
            std *= 1.0/(i-1);
            std = sqrt(std);
            
            print("batch size = '${BS}'");
            print("input size = '${IS}'");
            print("avg latency (s) = " avg );
            print("latency std (s) = " std );
            print("latency interval 50% (s) = " l50);
            print("latency interval 90% (s) = " l90);
            print("latency interval 95% (s) = " l95);
            print("latency interval 99% (s) = " l99);
            print("average mels generated = " mels);
            print("average audio generated (s) = " as);
            print("average real-time factor = " rtf);
         }'
rm "${INPUT}"

