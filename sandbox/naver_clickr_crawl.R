############################################################
######### R code for crawling Naver trends data ############
############################################################
## click rates for 11 categories

# import library
library(readxl)
library(openxlsx)
library(stringr)
library(dplyr)
library(ggplot2)
library(reshape)
library(stringr)
library(httr)
#install.packages("XML", repos = "http://www.omegahat.net/R") 
library(XML) 
library(jsonlite)


api_url = "https://openapi.naver.com/v1/search/shop.xml"

query = URLencode(iconv("안드로이드", to="UTF-8"))
query = str_c("?query=", query)
client_id = '2lxwbBf9CLh83NXM9pkH'
client_secret = 'zPtWdrmCUt'
result = GET(str_c(api_url, query), 
             add_headers("X-Naver-Client-Id" = client_id, "X-Naver-Client-Secret" = client_secret))
xml_ = xmlParse(result)
xpathSApply(xml_, "/rss/channel/item/title", xmlValue)
xpathSApply(xml_, "/rss/channel/item/link", xmlValue)
xpathSApply(xml_, "/rss/channel/item/description", xmlValue)



result = GET("https://openapi.naver.com/v1/search/shop.xml?query=%EC%A3%BC%EC%8B%9D&display=10&start=1&sort=sim", 
             add_headers("X-Naver-Client-Id" = client_id, "X-Naver-Client-Secret" = client_secret))
xml_ = xmlParse(result)
xpathSApply(xml_, "/rss/channel/item/title", xmlValue)
xpathSApply(xml_, "/rss/channel/item/link", xmlValue)
xpathSApply(xml_, "/rss/channel/item/description", xmlValue)


# 11 categories/ 4trials
# first trial
tmp = fromJSON('../data/naver_shopping_click_rate/naver_shopping_1.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,19),547) %>% as.data.frame(),
      substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
      substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,10),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp4 = cbind(rep(substr(tmp[3,1],7,12),547) %>% as.data.frame(),
             substr(tmp[3,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[3,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp4) = c('category','date','click_rate')
levels(tmp4[,2])[547] = '2019-01-01'

tmp_1 = rbind(tmp2,tmp3,tmp4)

# second trial
tmp = fromJSON('../data/naver_shopping_click_rate/naver_shopping_2.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,21),547) %>% as.data.frame(),
             substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,13),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp4 = cbind(rep(substr(tmp[3,1],7,11),547) %>% as.data.frame(),
             substr(tmp[3,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[3,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp4) = c('category','date','click_rate')
levels(tmp4[,2])[547] = '2019-01-01'

tmp_2 = rbind(tmp2,tmp3,tmp4)


# third trial
tmp = fromJSON('../data/naver_shopping_click_rate/naver_shopping_3.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,17),547) %>% as.data.frame(),
             substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,13),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp4 = cbind(rep(substr(tmp[3,1],7,11),547) %>% as.data.frame(),
             substr(tmp[3,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[3,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp4) = c('category','date','click_rate')
levels(tmp4[,2])[547] = '2019-01-01'

tmp_3 = rbind(tmp2,tmp3,tmp4)



# fourth trial
tmp = fromJSON('../data/naver_shopping_click_rate/naver_shopping_4.json')
tmp = gsub('\\"','',tmp) %>% 
  gsub('\\{','',.) %>% 
  gsub('\\}','',.) %>% 
  str_split(., ',') %>% 
  unlist() %>%
  .[-c(1:3)] %>% 
  matrix(., ncol=1096,byrow = TRUE)

tmp2 = cbind(rep(substr(tmp[1,1],16,22),547) %>% as.data.frame(),
             substr(tmp[1,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[1,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp2) = c('category','date','click_rate')
levels(tmp2[,2])[547] = '2019-01-01'

tmp3 = cbind(rep(substr(tmp[2,1],7,9),547) %>% as.data.frame(),
             substr(tmp[2,seq(3, ncol(tmp), by=2)], 8,17) %>% as.data.frame(),
             substr(tmp[2,seq(4, ncol(tmp), by=2)], 7,12) %>% as.data.frame()) %>% 
  as.data.frame()
colnames(tmp3) = c('category','date','click_rate')
levels(tmp3[,2])[547] = '2019-01-01'

tmp_4 = rbind(tmp2,tmp3)
tmp = rbind(tmp_1,tmp_2,tmp_3,tmp_4) #crawled data from naver trends








