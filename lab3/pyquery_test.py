from pyquery import PyQuery as pq
from lxml import etree
import urllib
from rdflib import URIRef, BNode, Literal, Graph, Namespace, term
from rdflib.namespace import RDF, FOAF, XSD
from string import Template
from datetime import datetime

g = Graph()
n = Namespace("https://www.chalmers.se/en/about-chalmers/calendar/Pages/default.aspx/")

d = pq(url="https://www.chalmers.se/en/about-chalmers/calendar/Pages/default.aspx")
div_items_list = pq(d).find('.feed-list').find('li').find('div').filter('.item')
for item in div_items_list:
    
    dates = pq(item).find('div').filter('.left-col.calendar').find('span')
    month = pq(dates).filter('.month').text()
    year = pq(dates).filter('.year').text()
    day = pq(dates).filter('.day').text()
    time = pq(pq(item).find('div').filter('.middle-col.desc.hidden-xs').find('span').filter('.meta')[1]).text().replace(' Time: ', '')
    datestr = '{}-{}-{} {}'.format(year,month,day,time)

    description = pq(item).find('div').find('span').filter('.desc').text()
    link = pq(item).find('div').filter('.middle-col.desc.hidden-xs').find('a').attr('href')
    title = pq(item).find('div').filter('.middle-col.desc.hidden-xs').find('a').filter('.title').text()
    type = pq(pq(item).find('div').filter('.middle-col.desc.hidden-xs').find('span').filter('.meta').outerHtml()).text()
    event = URIRef(link)
    dt = datetime.strptime(datestr.lower() , '%Y-%b-%d %H:%M')

    g.add((event, FOAF.title, Literal(title)))
    g.add((event, n.term('description'), Literal(description)))
    g.add((event, n.term('date'), Literal(dt, datatype=XSD.date)))
    g.add((event, n.term('type'), Literal(type)))
    
print(g.serialize(format='application/rdf+xml'))