#include "include/int2048.h"

namespace sjtu {

class BigIntCore {
public:
  static const int base = 10000; // 1e4
  static const int base_digits = 4;
};

static void trim(std::vector<int> &a) { while (!a.empty() && a.back() == 0) a.pop_back(); }

static int cmp_abs(const std::vector<int> &a, const std::vector<int> &b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (int i = (int)a.size() - 1; i >= 0; --i) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}

static std::vector<int> add_abs(const std::vector<int> &a, const std::vector<int> &b) {
  const int base = BigIntCore::base;
  std::vector<int> res; res.reserve(std::max(a.size(), b.size()) + 1);
  int carry = 0; size_t n = std::max(a.size(), b.size());
  for (size_t i = 0; i < n || carry; ++i) {
    long long cur = carry;
    if (i < a.size()) cur += a[i];
    if (i < b.size()) cur += b[i];
    res.push_back((int)(cur % base));
    carry = (int)(cur / base);
  }
  return res;
}

static std::vector<int> sub_abs(const std::vector<int> &a, const std::vector<int> &b) {
  const int base = BigIntCore::base;
  std::vector<int> res; res.reserve(a.size());
  int carry = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    long long cur = a[i] - carry - (i < b.size() ? b[i] : 0);
    if (cur < 0) { cur += base; carry = 1; } else carry = 0;
    res.push_back((int)cur);
  }
  trim(res); return res;
}

struct C { long double x, y; C(long double X=0, long double Y=0):x(X),y(Y){} };
static C operator+(const C &a, const C &b){ return C(a.x+b.x, a.y+b.y);}
static C operator-(const C &a, const C &b){ return C(a.x-b.x, a.y-b.y);}
static C operator*(const C &a, const C &b){ return C(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);}

static void fft(std::vector<C> &a, bool invert) {
  int n = (int)a.size();
  for (int i=1, j=0; i<n; ++i) { int bit = n>>1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i<j) std::swap(a[i], a[j]); }
  for (int len=2; len<=n; len<<=1) {
    long double ang = 2.0L * 3.14159265358979323846264338327950288L / len * (invert ? -1.0L : 1.0L);
    C wlen(std::cos(ang), std::sin(ang));
    for (int i=0; i<n; i+=len) {
      C w(1,0);
      for (int j=0; j<len/2; ++j) { C u = a[i+j]; C v = a[i+j+len/2] * w; a[i+j] = u + v; a[i+j+len/2] = u - v; w = w * wlen; }
    }
  }
  if (invert) for (int i=0;i<n;++i) { a[i].x /= n; a[i].y /= n; }
}

static std::vector<int> mul_abs_fft(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<C> fa, fb; int n = 1;
  while (n < (int)a.size() + (int)b.size()) n <<= 1;
  fa.resize(n); fb.resize(n);
  for (size_t i=0;i<a.size();++i) fa[i].x = a[i]; for (size_t i=a.size(); i<(size_t)n; ++i) fa[i] = C();
  for (size_t i=0;i<b.size();++i) fb[i].x = b[i]; for (size_t i=b.size(); i<(size_t)n; ++i) fb[i] = C();
  fft(fa,false); fft(fb,false);
  for (int i=0;i<n;++i) fa[i] = fa[i]*fb[i];
  fft(fa,true);
  std::vector<int> res(n); long long carry = 0; const int base = BigIntCore::base;
  for (int i=0;i<n;++i) { long long t = (long long)std::llround(fa[i].x) + carry; res[i] = (int)(t % base); carry = t / base; }
  while (carry) { res.push_back((int)(carry % base)); carry /= base; }
  trim(res); return res;
}

// helper for division in big-endian representation
static int cmp_abs_be(const std::vector<int> &A, const std::vector<int> &B) {
  if (A.size() != B.size()) return A.size() < B.size() ? -1 : 1;
  for (size_t i=0;i<A.size();++i) if (A[i] != B[i]) return A[i] < B[i] ? -1 : 1;
  return 0;
}

static std::vector<int> mul_abs_be_digit(const std::vector<int> &B, int k) {
  const int base = BigIntCore::base; std::vector<int> R(B.size()); long long carry = 0;
  for (int i=(int)B.size()-1; i>=0; --i) { long long cur = (long long)B[i]*k + carry; R[i] = (int)(cur % base); carry = cur / base; }
  if (carry) R.insert(R.begin(), (int)carry);
  size_t p=0; while (p<R.size() && R[p]==0) ++p; if (p) R.erase(R.begin(), R.begin()+p);
  return R;
}

static std::vector<int> sub_abs_be(const std::vector<int> &A, const std::vector<int> &B) {
  const int base = BigIntCore::base; int n = (int)std::max(A.size(), B.size()); std::vector<int> R(n,0);
  int ia = (int)A.size()-1, ib = (int)B.size()-1, ir = n-1; long long borrow = 0;
  for (; ir>=0; --ir) { long long av = (ia>=0 ? A[ia] : 0); long long bv = (ib>=0 ? B[ib] : 0); long long cur = av - bv - borrow; if (cur < 0) { cur += base; borrow = 1; } else borrow = 0; R[ir] = (int)cur; --ia; --ib; }
  size_t p=0; while (p<R.size() && R[p]==0) ++p; if (p) R.erase(R.begin(), R.begin()+p);
  return R;
}

static void div_mod_abs(const std::vector<int> &a_le, const std::vector<int> &b_le, std::vector<int> &q_le, std::vector<int> &r_le) {
  q_le.clear(); r_le.clear(); if (b_le.empty()) return; if (cmp_abs(a_le, b_le) < 0) { r_le = a_le; return; }
  std::vector<int> A(a_le.rbegin(), a_le.rend()); std::vector<int> B(b_le.rbegin(), b_le.rend());
  std::vector<int> R; std::vector<int> Q; Q.reserve(A.size()); const int base = BigIntCore::base;
  for (size_t idx=0; idx<A.size(); ++idx) {
    if (!R.empty() || A[idx]!=0) R.push_back(A[idx]);
    if (R.empty()) { Q.push_back(0); continue; }
    int lo = 0, hi = base - 1, best = 0;
    while (lo <= hi) { int mid = (lo + hi) >> 1; std::vector<int> M = mul_abs_be_digit(B, mid); int cmp = cmp_abs_be(M, R); if (cmp <= 0) { best = mid; lo = mid + 1; } else { hi = mid - 1; } }
    if (best) { std::vector<int> M = mul_abs_be_digit(B, best); R = sub_abs_be(R, M); }
    size_t p=0; while (p<R.size() && R[p]==0) ++p; if (p) R.erase(R.begin(), R.begin()+p);
    Q.push_back(best);
  }
  size_t p=0; while (p<Q.size() && Q[p]==0) ++p; if (p) Q.erase(Q.begin(), Q.begin()+p);
  q_le.assign(Q.rbegin(), Q.rend()); trim(q_le); r_le.assign(R.rbegin(), R.rend()); trim(r_le);
}

// Constructors
int2048::int2048() : neg(false), d() {}

int2048::int2048(long long v) : neg(v<0), d() {
  unsigned long long u = neg ? (unsigned long long)(-(v + 1)) + 1ULL : (unsigned long long)v;
  const int base = BigIntCore::base; while (u) { d.push_back((int)(u % base)); u /= base; }
}

int2048::int2048(const std::string &s) : neg(false), d() { read(s); }

int2048::int2048(const int2048 &o) : neg(o.neg), d(o.d) {}

void int2048::read(const std::string &s) {
  neg=false; d.clear(); int pos = 0; while (pos<(int)s.size() && (s[pos]==' '||s[pos]=='\n' || s[pos]=='\t')) ++pos;
  if (pos<(int)s.size() && (s[pos]=='+'||s[pos]=='-')) { neg = (s[pos]=='-'); ++pos; }
  while (pos<(int)s.size() && s[pos]=='0') ++pos;
  std::vector<int> tmp; const int bd = BigIntCore::base_digits;
  for (int i = (int)s.size(); i > pos; i -= bd) {
    int x = 0; int l = std::max(pos, i - bd); for (int j=l; j<i; ++j) x = x*10 + (s[j]-'0'); tmp.push_back(x);
  }
  d = tmp; trim(d); if (d.empty()) neg=false;
}

void int2048::print() {
  if (d.empty()) { std::cout << 0; return; }
  if (neg) std::cout << '-'; int bd = BigIntCore::base_digits; std::cout << d.back();
  for (int i=(int)d.size()-2; i>=0; --i) { char buf[16]; std::snprintf(buf, sizeof(buf), "%0*d", bd, d[i]); std::cout << buf; }
}

int2048 &int2048::add(const int2048 &b) {
  if (neg == b.neg) { d = add_abs(d, b.d); }
  else { int c = cmp_abs(d, b.d); if (c >= 0) { d = sub_abs(d, b.d); } else { d = sub_abs(b.d, d); neg = b.neg; } }
  if (d.empty()) neg=false; return *this;
}

int2048 add(int2048 a, const int2048 &b) { a.add(b); return a; }

int2048 &int2048::minus(const int2048 &b) {
  if (neg != b.neg) { d = add_abs(d, b.d); }
  else { int c = cmp_abs(d, b.d); if (c >= 0) { d = sub_abs(d, b.d); } else { d = sub_abs(b.d, d); neg = !neg; } }
  if (d.empty()) neg=false; return *this;
}

int2048 minus(int2048 a, const int2048 &b) { a.minus(b); return a; }

int2048 int2048::operator+() const { return *this; }

int2048 int2048::operator-() const { int2048 r(*this); if (!r.d.empty()) r.neg = !r.neg; return r; }

int2048 &int2048::operator=(const int2048 &o) { if (this == &o) return *this; neg = o.neg; d = o.d; return *this; }

int2048 &int2048::operator+=(const int2048 &b) { return add(b); }
int2048 operator+(int2048 a, const int2048 &b) { return a += b; }

int2048 &int2048::operator-=(const int2048 &b) { return minus(b); }
int2048 operator-(int2048 a, const int2048 &b) { return a -= b; }

int2048 &int2048::operator*=(const int2048 &b) {
  if (d.empty() || b.d.empty()) { d.clear(); neg=false; return *this; }
  std::vector<int> res; size_t n = d.size(), m = b.d.size();
  if (std::min(n,m) < 64) {
    const int base = BigIntCore::base; res.assign(n + m, 0);
    for (size_t i=0;i<n;++i) { long long carry = 0; for (size_t j=0;j<m || carry;++j) {
      long long cur = res[i+j] + (long long)d[i] * (j<m ? b.d[j] : 0) + carry; res[i+j] = (int)(cur % base); carry = cur / base; } }
    trim(res);
  } else { res = mul_abs_fft(d, b.d); }
  d.swap(res); neg = (neg != b.neg); if (d.empty()) neg=false; return *this;
}

int2048 operator*(int2048 a, const int2048 &b) { return a *= b; }

int2048 &int2048::operator/=(const int2048 &b) {
  if (b.d.empty()) return *this; if (d.empty()) { neg=false; return *this; }
  std::vector<int> q, r; div_mod_abs(d, b.d, q, r);
  bool negQ = (neg != b.neg);
  if (!r.empty() && negQ) {
    const int base = BigIntCore::base; int carry = 1;
    for (size_t i=0; i<q.size() && carry; ++i) { int cur = q[i] + carry; if (cur >= base) { q[i] = cur - base; carry = 1; } else { q[i] = cur; carry = 0; } }
    if (carry) q.push_back(carry);
  }
  d.swap(q); neg = (!d.empty() ? negQ : false); return *this;
}

int2048 operator/(int2048 a, const int2048 &b) { return a /= b; }

int2048 &int2048::operator%=(const int2048 &b) {
  if (b.d.empty()) return *this; if (d.empty()) { neg=false; return *this; }
  std::vector<int> q, r; div_mod_abs(d, b.d, q, r);
  if (!r.empty() && (neg != b.neg)) { r = sub_abs(b.d, r); }
  d.swap(r); neg = (!d.empty() ? b.neg : false); return *this;
}

int2048 operator%(int2048 a, const int2048 &b) { return a %= b; }

std::istream &operator>>(std::istream &in, int2048 &x) { std::string s; in >> s; x.read(s); return in; }

std::ostream &operator<<(std::ostream &out, const int2048 &x) {
  if (x.d.empty()) { out << 0; return out; }
  if (x.neg) out << '-'; int bd = BigIntCore::base_digits; out << x.d.back();
  for (int i=(int)x.d.size()-2; i>=0; --i) { char buf[16]; std::snprintf(buf, sizeof(buf), "%0*d", bd, x.d[i]); out << buf; }
  return out;
}

bool operator==(const int2048 &a, const int2048 &b) { return a.neg==b.neg && a.d==b.d; }
bool operator!=(const int2048 &a, const int2048 &b) { return !(a==b); }

bool operator<(const int2048 &a, const int2048 &b) {
  if (a.neg != b.neg) return a.neg; int c = cmp_abs(a.d, b.d); return a.neg ? (c>0) : (c<0);
}

bool operator>(const int2048 &a, const int2048 &b) { return b < a; }
bool operator<=(const int2048 &a, const int2048 &b) { return !(b<a); }
bool operator>=(const int2048 &a, const int2048 &b) { return !(a<b); }

} // namespace sjtu
