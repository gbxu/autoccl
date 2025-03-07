// clang-format off
#pragma once
#include "src/include/internal/logging.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <string>

#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
#define MAX_IF_NAME_SIZE 16
#define MAX_IFS 16
/* Common socket address storage structure for IPv4/IPv6 */
union SocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c; // NOHINT
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

/* Format a string representation of a (union SocketAddress *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char *SocketToString(union SocketAddress *addr, char *buf, const int numericHostForm=1) {
  if (buf == NULL || addr == NULL) return NULL;
  struct sockaddr *saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  (void) getnameinfo(saddr, sizeof(union SocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag);
  sprintf(buf, "%s:%s", host, service);
  return buf;
}

int SocketGetAddrFromString(union SocketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    THROWERROR << "Net : string is null";
    return -1;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      THROWERROR << "Net : No valid <IPv4_or_hostname>:<port> pair found";
      return -1;
    }

    struct addrinfo hints, *p; // NOHINT
    int rv; // NOHINT
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      THROWERROR << "Net : error encountered when getting address info : " << gai_strerror(rv);
      return -1;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      THROWERROR << "Net : unsupported IP family";
      return -1;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair); // NOHINT
    for (i = 1; i < len; i++) { // NOHINT
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      THROWERROR << "Net : No valid [IPv6]:port pair found";
      return -1;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);
    strncpy(port_str, ip_port_pair+i+2, len-i-1);
    int port = atoi(port_str);
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return 0;
}

bool matchSubnet(struct ifaddrs local_if, union SocketAddress* remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr); // NOHINT
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask); // NOHINT
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr); // NOHINT
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask); // NOHINT
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;  //IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  //Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    THROWERROR << "Net : Unsupported address family type";
    return false;
  }
}

int FindInterfaceMatchSubnet(char* ifNames, union SocketAddress* localAddrs, union SocketAddress* coordinatorAddr, int ifNameMaxSize, int maxIfs) {
  char line_a[SOCKET_NAME_MAXLEN+1];
  int found = 0;
  struct ifaddrs *interfaces, *interface; // NOHINT
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, coordinatorAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memcpy(localAddrs+found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames+found*ifNameMaxSize, interface->ifa_name, ifNameMaxSize);

    found++;
    if (found == maxIfs) break;
  }

  if (found == 0) {
    WARN(Logger::LogSubSys::NET) << "Net : No interface found in the same subnet as remote address " << SocketToString(coordinatorAddr, line_a);
  }
  freeifaddrs(interfaces);
  return found;
}

int getAvailablePort(SocketAddress& addr) {
  /* IPv4/IPv6 support */
  int family = addr.sa.sa_family;
  if (family != AF_INET && family != AF_INET6) {
    return -1;
  }
  socklen_t salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);

  /* Connect to a hostname / port */
  int fd = socket(family, SOCK_STREAM, 0);
  if (fd == -1) {
    return -1;
  }

  if (family == AF_INET) {
    addr.sin.sin_port = htons(0);
  } else {
    addr.sin6.sin6_port = htons(0);
  }

  if (bind(fd, &addr.sa, salen) != 0) {
    close(fd);
    return -1;
  }

  if (getsockname(fd, &addr.sa, &salen) != 0) {
    close(fd);
    return -1;
  }
  int ret_port = ntohs(family == AF_INET ? addr.sin.sin_port : addr.sin6.sin6_port);
  close(fd);
  return ret_port;
}

static bool matchIf(const char* string, const char* ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i=0; i<listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact)
        && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

static int findInterfaces(const char* prefixList, char* names, union SocketAddress *addrs, int sock_family, int maxIfNameSize, int maxIfs) {
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names+i*maxIfNameSize) == 0) { duplicate = true; break; }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names+found*maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memcpy(addrs+found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  char* env = getenv("SOCKET_FAMILY");
  if (env == NULL)
    return family;

  INFO(Logger::LogSubSys::NET) << "SOCKET_FAMILY set by environment to" << env;

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

int tryFindInterfaces(char* ifNames, union SocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs) {
  static int shownIfName = 0;
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  char* env = getenv("NCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    printf("NCCL_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0) printf("NCCL_SOCKET_IFNAME set to %s", env);
    nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0) nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0) nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    if (nIfs == 0) nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  }
  return nIfs;
}

std::string getMatchInterfaceIpPort(const char* env) {
  char ifName[MAX_IF_NAME_SIZE+1];
  union SocketAddress ifAddr;

  union SocketAddress coordinatorAddr;
  if (SocketGetAddrFromString(&coordinatorAddr, env) != 0) {
    THROWERROR << "Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>";
  }
  if (FindInterfaceMatchSubnet(ifName, &ifAddr, &coordinatorAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
    int nIfs = tryFindInterfaces(ifName, &ifAddr, MAX_IF_NAME_SIZE, 1);
    if (nIfs <= 0) {
      THROWERROR << "NET/Socket : No usable listening interface found.";
    }    
  }
  char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
  sprintf(line, " %s:", ifName);
  getAvailablePort(ifAddr);
  SocketToString(&ifAddr, line+strlen(line));
  return std::string(line);
}
// clang-format on
