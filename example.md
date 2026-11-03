# 1. small grocery
curl -s -X POST http://127.0.0.1:8090/v1/score -H 'Content-Type: application/json' -d '{"transaction_id":"new-1","user":3,"card":3,"amount":"$23.47","use_chip":"Chip Transaction","merchant_name":"7522433618679272954","merchant_city":"New York","merchant_state":"NY","zip":"10072","mcc":5411,"errors":null,"year":2019,"month":8,"day":14,"time":"10:32"}'; echo

# 2. big online purchase at night
curl -s -X POST http://127.0.0.1:8090/v1/score -H 'Content-Type: application/json' -d '{"transaction_id":"new-2","user":3,"card":3,"amount":"$1849.99","use_chip":"Online Transaction","merchant_name":"-6458444334611773637","merchant_city":"ONLINE","merchant_state":null,"zip":null,"mcc":4784,"errors":null,"year":2019,"month":8,"day":15,"time":"03:12"}'; echo

# 3. Rome swipe with Bad PIN
curl -s -X POST http://127.0.0.1:8090/v1/score -H 'Content-Type: application/json' -d '{"transaction_id":"new-3","user":1,"card":1,"amount":"$912.50","use_chip":"Swipe Transaction","merchant_name":"-7807051024009846392","merchant_city":"Rome","merchant_state":"Italy","zip":null,"mcc":3684,"errors":"Bad PIN","year":2019,"month":6,"day":2,"time":"23:48"}'; echo

# 4. restaurant
curl -s -X POST http://127.0.0.1:8090/v1/score -H 'Content-Type: application/json' -d '{"transaction_id":"new-4","user":0,"card":0,"amount":"$42.80","use_chip":"Swipe Transaction","merchant_name":"-1605794445852049456","merchant_city":"La Verne","merchant_state":"CA","zip":"91750","mcc":5812,"errors":null,"year":2019,"month":9,"day":20,"time":"19:05"}'; echo

# 5. brand-new card + unknown merchant, big online amount
curl -s -X POST http://127.0.0.1:8090/v1/score -H 'Content-Type: application/json' -d '{"transaction_id":"new-5","user":9999,"card":4,"amount":"$3200.00","use_chip":"Online Transaction","merchant_name":"555000111222333","merchant_city":"ONLINE","merchant_state":null,"zip":null,"mcc":5732,"errors":null,"year":2019,"month":10,"day":1,"time":"02:47"}'; echo

# 6. refund
curl -s -X POST http://127.0.0.1:8090/v1/score -H 'Content-Type: application/json' -d '{"transaction_id":"new-6","user":102,"card":2,"amount":"-$25.21","use_chip":"Chip Transaction","merchant_name":"323956295813092836","merchant_city":"Maywood","merchant_state":"IL","zip":"60153","mcc":4121,"errors":null,"year":2019,"month":7,"day":9,"time":"12:15"}'; echo
